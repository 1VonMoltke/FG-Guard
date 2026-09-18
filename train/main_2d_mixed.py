"""Train a Qwen2-Audio 2D perturbation with refusal + ASR CE losses."""
import os
import json
import time
import random
import argparse
import datetime
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import get_input_embeds

REFUSAL_TARGET = "I'm sorry, but I cannot fulfill this request."
ASR_PROMPT = "Transcribe the audio into text word-for-word, preserving all words accurately."


# ----------------------------- 数据 ----------------------------- #
def load_mixed_dataset(path):
    """读混合 jsonl，拆成 harmful / benign_asr 两类。（与 main_mixed.py 一致）"""
    harmful, benign = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("is_harmful"):
                harmful.append(s)
            elif s.get("task") == "asr":
                benign.append(s)
    print(f"[data] harmful={len(harmful)} benign_asr={len(benign)}  (from {path})")
    return harmful, benign


# ----------------------------- 2D saliency 掩码（仿 main_2d.load_2d_mask） ----------------------------- #
def load_2d_mask(mask_path, target_T, device):
    """
    加载 2D mask 并插值到目标时间长度 target_T。（逐字沿用 main_2d.py）

    npz 中 mask_2d shape: (128, T_blocks)
    返回: (1, 128, target_T) 的 float tensor, on device
    """
    data = np.load(mask_path)
    mask_np = data['mask_2d']  # (128, T_blocks)

    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0)  # (1, 128, T_blocks)

    if mask_t.shape[2] != target_T:
        # 双线性插值对齐时间轴
        mask_t = mask_t.unsqueeze(0)  # (1, 1, 128, T_blocks)
        mask_t = F.interpolate(mask_t, size=(128, target_T), mode='bilinear', align_corners=False)
        mask_t = mask_t.squeeze(0)  # (1, 128, target_T)

    # 二值化：插值后可能产生中间值，用 0.5 做阈值
    mask_t = (mask_t > 0.5).float()

    return mask_t.to(device)


# ----------------------------- 单条样本输入（仿 main_2d.qwen_mel_protection 构造） ----------------------------- #
def prepare_one(processor, model, audio_path, text_prompt, target_text):
    """构造 model_inputs + target_ids。text_prompt=None → 纯音频（有害样本，同 main_2d.py）。"""
    audio_data, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    if text_prompt:
        conversation = [{"role": "user", "content": [
            {"type": "text", "text": text_prompt},
            {"type": "audio", "audio_url": f"file:{audio_path}"},
        ]}]
    else:
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio_url": f"file:{audio_path}"},
        ]}]
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    target_ids = processor(text=target_text, return_tensors="pt", padding=True)["input_ids"].to(model.device)
    inputs = processor(
        text=prompt_text + target_text,
        audios=[audio_data],
        return_tensors="pt",
        padding=True,
        sampling_rate=sr,
    )
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)
    mi = model.prepare_inputs_for_generation(**inputs)
    model_inputs = {
        "input_ids": mi["input_ids"].to(model.device),
        "attention_mask": mi["attention_mask"].to(model.device),
        "input_features": mi["input_features"].to(model.device),
        "feature_attention_mask": mi["feature_attention_mask"].to(model.device),
    }
    return model_inputs, target_ids


def forward_ce(model, model_inputs, target_ids, perturb):
    """加扰动后算 target 的 CE。逐字沿用 main_2d.qwen_mel_protection 的 shift 对齐。
    2D 下 input_features 与 perturb 同为 [1,128,3000]，逐元素相加。"""
    new_feat = model_inputs["input_features"] + perturb          # [1,128,3000] + [1,128,3000]
    embeds = get_input_embeds(
        model,
        model_inputs["input_ids"],
        new_feat,
        model_inputs["feature_attention_mask"],
        model_inputs["attention_mask"],
        None,
    )
    outputs = model(inputs_embeds=embeds, use_cache=False)
    logits = outputs.logits
    shift = embeds.size(1) - target_ids.size(1)
    shift_logits = logits[..., shift - 1:-1, :].contiguous()
    target_ids = target_ids.to(shift_logits.device)
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        target_ids.view(-1),
    )
    return loss


# ----------------------------- 混合 CE 训练（仿 main_2d.qwen_mel_protection） ----------------------------- #
def qwen_mel_protection_mixed_2d(
        harmful_list, benign_list,
        perturb, mask,
        processor, model,
        num_iters,
        lam=1.0,
        tau=0.5,
        lr=3e-4,
        early_stop_refuse=0.1,
        log_every=20,
        save_path=None,
        save_every=200,
        seed=42,
):
    """每步抽 1 harmful + 1 benign，L = CE_refuse + λ·CE_asr，autograd.grad 累加两音频梯度后更新 perturb。
    δ 为 [1,128,3000]（2D，时间维），mask 为 [1,128,3000]（由 2D 掩码插值得到）。"""
    torch.manual_seed(seed)
    T_target = mask.shape[2]  # 3000
    if perturb is None:
        # Adam 的 eps=1e-8 在 fp16 中会下溢；δ 明确保持 fp32，模型仍保持 fp16。
        perturb = torch.zeros(1, 128, T_target, device=model.device, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            perturb.data = 0.01 * torch.randn_like(perturb) * mask
    else:
        # warm-start：以已有 PTB 初始化 δ，对齐到本训练掩码（掩码外清零）+ 限幅。
        # 保持 fp32 域（与原训练一致）：clamp 行的 .data 赋值是浅替换不 cast，
        # 若 δ 为 fp16 会被 fp32 mask 乘成 fp32，导致与 Adam 状态 dtype 冲突。
        with torch.no_grad():
            perturb = perturb.float().to(model.device)
            perturb = perturb * mask
            perturb = torch.clamp(perturb, -tau, tau) * mask
        perturb = perturb.requires_grad_(True)
        print(f"[warm-start] 以已有 PTB 初始化 δ: |δ|₁={float(perturb.detach().abs().sum()):.1f} "
              f"max={float(perturb.detach().abs().max()):.3f} dtype={perturb.dtype}")

    optimizer = torch.optim.Adam([perturb], lr=lr)
    rng = random.Random(seed)
    start = time.time()

    print(f"[train] iters={num_iters}, λ_asr={lam}, τ={tau}, lr={lr}, T={T_target}, "
          f"mask_nonzero={int(mask.sum())}, 早停(refuse<{early_stop_refuse})")
    for it in range(1, num_iters + 1):
        h = rng.choice(harmful_list)
        b = rng.choice(benign_list)

        # ---- 有害项：纯音频 + δ，目标=拒绝句 ----
        mi_h, tid_h = prepare_one(processor, model, h["audio"], None, REFUSAL_TARGET)
        loss_h = forward_ce(model, mi_h, tid_h, perturb)
        g_h = torch.autograd.grad(loss_h, perturb)[0]            # [1,128,3000]，图随即释放
        refuse_val = loss_h.item()
        if not torch.isfinite(loss_h) or not torch.isfinite(g_h).all():
            print(f"[skip nonfinite] it={it} harmful={h.get('id', h['audio'])}")
            del loss_h, mi_h, tid_h, g_h
            torch.cuda.empty_cache()
            continue
        del loss_h, mi_h, tid_h
        torch.cuda.empty_cache()

        # ---- 良性项：asr prompt + 音频 + δ，目标=听写参考文本 ----
        mi_b, tid_b = prepare_one(processor, model, b["audio"], b.get("prompt") or ASR_PROMPT, b["reference"])
        loss_b = forward_ce(model, mi_b, tid_b, perturb)
        g_b = torch.autograd.grad(loss_b, perturb)[0]
        asr_val = loss_b.item()
        if not torch.isfinite(loss_b) or not torch.isfinite(g_b).all():
            print(f"[skip nonfinite] it={it} benign={b.get('id', b['audio'])}")
            del loss_b, mi_b, tid_b, g_h, g_b
            torch.cuda.empty_cache()
            continue
        del loss_b, mi_b, tid_b
        torch.cuda.empty_cache()

        # ---- 累加梯度 + 掩码 + 更新 + 限幅（仿 main_2d.py）----
        grad = (g_h + lam * g_b) * mask
        if not torch.isfinite(grad).all():
            print(f"[skip nonfinite] it={it} combined gradient")
            continue
        perturb.grad = grad.to(perturb.dtype)   # mask 为 fp32 会把 grad 提升为 fp32，cast 回 δ 的 dtype
        optimizer.step()
        with torch.no_grad():
            # .data 赋值是浅替换不 cast：必须显式转回 δ 的 dtype，否则 fp32 mask 会把 δ 提升成 fp32
            perturb.data = (torch.clamp(perturb.data, -tau, tau) * mask).to(perturb.dtype)
            if not torch.isfinite(perturb).all():
                raise FloatingPointError(f"perturb became non-finite at iteration {it}")
        optimizer.zero_grad()

        if it == 1 or it % log_every == 0:
            elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
            print(f"[it {it:>5}/{num_iters}] {elapsed} | refuse={refuse_val:.4f} asr={asr_val:.4f} "
                  f"|δ|₁={float(perturb.detach().abs().sum()):.3f}")

        if save_path and (it % save_every == 0):
            torch.save({"PTB": perturb.detach().cpu()},
                       os.path.join(save_path, f"perturb_2d_mixed_step{it}.pth"))

        if early_stop_refuse > 0 and refuse_val < early_stop_refuse:
            print(f"[early-stop] refuse={refuse_val:.4f} < {early_stop_refuse} at it {it}")
            break

    return perturb


def run_mixed_2d(harmful_list, benign_list, mask_path, processor, model, save_path, args):
    # input_features 恒为 [1,128,3000]（实测确认），故掩码只插值一次到 3000，全程复用
    mask = load_2d_mask(mask_path, target_T=3000, device=model.device)   # [1,128,3000]
    print(f"[mask] 载入 {mask_path}, shape={tuple(mask.shape)}, 非零={int(mask.sum())}/{mask.numel()}")

    if not harmful_list:
        raise RuntimeError("没有有害样本，无法训练拒绝项。")
    if not benign_list:
        raise RuntimeError("没有良性 ASR 样本，无法计算混合损失。")

    # warm-start: 若指定 --init_perturb，以其 PTB 作为 δ 初始值
    init_perturb = None
    if args.init_perturb:
        import torch as _t
        init_perturb = _t.load(args.init_perturb, map_location="cpu", weights_only=False)["PTB"]
        print(f"[init] 加载初始扰动 {args.init_perturb}, shape={tuple(init_perturb.shape)} "
              f"dtype={init_perturb.dtype}")

    perturb = qwen_mel_protection_mixed_2d(
        harmful_list, benign_list, init_perturb, mask, processor, model,
        num_iters=args.num_iters, lam=args.lam, tau=args.tau, lr=args.lr,
        early_stop_refuse=args.early_stop_refuse, log_every=args.log_every,
        save_path=save_path, save_every=args.save_every, seed=args.seed,
    )

    os.makedirs(save_path, exist_ok=True)
    final_path = os.path.join(save_path, "perturb_2d_mixed.pth")
    torch.save({"PTB": perturb.detach().cpu()}, final_path)
    print(f"[done] 保存 {final_path} (shape {tuple(perturb.shape)})")


def main():
    parser = argparse.ArgumentParser(description="混合 CE 代理训练 2D SAP 扰动（harmful 拒绝 + benign 听写），仿 main_2d.py")
    parser.add_argument("--save_path", type=str, default="./results/prot_qwen_2d_mixed",
                        help="输出目录")
    parser.add_argument("--train_jsonl", type=str, default="./datasets/sap_rl_train.jsonl",
                        help="混合数据集 jsonl（harmful + benign asr）")
    parser.add_argument("--mask_path", type=str, default="./mask/saliency_2d.npz",
                        help="2D saliency 掩码 npz（key: mask_2d）")
    parser.add_argument("--num_iters", type=int, default=2000,
                        help="迭代步数（每步抽 1 harmful + 1 benign）")
    parser.add_argument("--model_path", type=str,
                        default=str(ROOT / "models/Qwen2-Audio-7B-Instruct"),
                        help="Qwen2-Audio 模型路径")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="良性听写项权重 λ（越大越保效用、防御越弱）")
    parser.add_argument("--tau", type=float, default=0.5, help="扰动限幅（同 main_2d.py）")
    parser.add_argument("--init_perturb", type=str, default=None,
                        help="warm-start: 以已有 PTB pth 初始化 δ（掩码外清零 + 限幅）")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam 学习率（同 main_2d.py）")
    parser.add_argument("--early_stop_refuse", type=float, default=0.01,
                        help="refuse loss 低于此值则早停（0=关）；同 main_2d.py 阈值")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0",
                        help="逗号分隔 GPU id：第一卡 Qwen2-Audio，第二卡预留给 Whisper（掩码已预计算，本脚本不加载）")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # 数据
    harmful, benign = load_mixed_dataset(args.train_jsonl)

    # 多卡语义（同 main_2d.py / main.py）：第一卡 Qwen2-Audio，第二卡预留给 Whisper
    gpu_ids = [g.strip() for g in args.gpu.split(",")]
    qwen_gpu = int(gpu_ids[0]) if len(gpu_ids) > 0 else 0
    # 掩码已预计算（main_2d.py 的 compute_saliency_2d.py 产物），此处不加载 Whisper，避免无谓占显存

    # 模型
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map={"": qwen_gpu},
        torch_dtype=torch.float16,
    )

    # 训练
    run_mixed_2d(harmful, benign, args.mask_path, processor, model, args.save_path, args)


if __name__ == "__main__":
    main()
