"""
Kimi-Audio adapter for ALMGuard (镜像 omni_adapter.py 的适配层设计).

让 ALMGuard 的 saliency / 扰动训练 / 评测 在 moonshotai/Kimi-Audio-7B-Instruct
（MoonshotKimiaForCausalLM）上运行。运行环境 = ALMGuard env（torch 2.2.2+cu121,
transformers 4.46.3），无需 flash-attn。

与 Qwen2-Audio 的差异（全部在官方 Kimi-Audio 推理代码里核对过，见 kimi-img/README.md）:
  - 模型为双流输入: input_ids(音频流) + text_input_ids(文本流) + is_continuous_mask
  - 音频不在模型内直接消费 mel，而是经 外部 WhisperEncoder(mel → (1,1500,1280))
    → 截断到 token_len*4 帧 → reshape (1, token_len, 5120)（每 4 个 encoder 帧 = 1 个 token，
    80ms/token），再由模型内部 vq_adaptor 投影到 hidden_size 并散播到
    media_begin/end 之间的 span 位置（continuous 位置乘 √2 与 token embedding 相加）。
  - 可微链: mel (1,128,3000) → whisper encoder(bf16, 冻结) → vq_adaptor → LLM → lm_head。
    因此 mel 仍是扰动空间，与 Qwen2-Audio 完全同构 (1,128,3000)。
  - 生成必须走官方手写循环（generation_mixin 不认识双流 forward）：prefill 全序列，
    之后每步喂 1 个新 token（音频流 blank、文本流采样文本），带 past_key_values。

本机环境关键适配:
  1. flash_attn 未安装，而 modeling 文件 import 时硬性检查 → 打桩 + 把
     MoonshotAttention._flash_attention_forward 替换为手动 attention。
     !!! 重要：不能用 torch 2.2.2 的 F.scaled_dot_product_attention(is_causal=True)
     做生成步（q_len=1 时因果掩码语义错误，把历史全部 mask 掉，模型输出退化为
     平分布逗号循环）。手动实现 flash 兼容的 offset 因果掩码（diagonal=kv_len-q_len）。
  2. whisper-large-v3/model.safetensors 缺少 __metadata__ 头，transformers 4.46.3
     from_pretrained 会崩 → 用 safetensors 手动加载 + load_state_dict。
  3. tokenizer 是 megatron 风格自定义实现（TikTokenTokenizer），不能用
     convert_tokens_to_ids / tokenizer(...) 等 transformers 标准 API：
     用 tokenizer.special_tokens[name] 取 id，tokenizer.encode(text, bos=, eos=) 编码，
     tokenizer.decode(ids) 解码。需 pip install blobfile（tiktoken 的纯文件读取依赖）。
"""

import math
import os
import sys
import types

import librosa
import numpy as np
import torch

# ---------------------------------------------------------------------------
# 环境修补（必须在 import modeling 之前调用）
# ---------------------------------------------------------------------------

def patch_flash_attn_requirement():
    """flash_attn 未安装时打桩，使 kimi 的 modeling 文件能 import。

    transformers.utils.is_flash_attn_2_available 被建模文件用于 import 分支
    （版本 >= 4.35 走 is_flash_attn_2_available）。打桩为 True + 注入假模块即可。
    真实调用路径随后被 _patch_eager_attention 替换，stub 函数永远不会被调用。
    """
    import transformers.utils as tu
    if tu.is_flash_attn_2_available():
        return  # flash_attn 已装，什么都不做（数值与官方一致）

    def _stub(*args, **kwargs):
        raise NotImplementedError("flash_attn 未安装，kimi_adapter 使用 eager 回退")

    fa = types.ModuleType("flash_attn")
    fa.flash_attn_func = _stub
    fa.flash_attn_varlen_func = _stub
    fb = types.ModuleType("flash_attn.bert_padding")
    fb.index_first_axis = _stub
    fb.pad_input = _stub
    fb.unpad_input = _stub
    sys.modules["flash_attn"] = fa
    sys.modules["flash_attn.bert_padding"] = fb
    tu.is_flash_attn_2_available = lambda: True


def _eager_flash_attention_forward(
    self, query_states, key_states, value_states, padding_mask,
    query_length, dropout=0.0, softmax_scale=None,
):
    """手动 attention，替换 MoonshotAttention._flash_attention_forward。

    与官方 flash_attn_func(..., causal=True) 语义一致：
      - q/k/v 传入形状 (bsz, q_len, n_heads, head_dim)
      - 因果掩码带生成偏移：query 第 i 行可见 key 列 j <= i + (kv_len - q_len)
        （prefill 时 q_len==kv_len → 严格下三角；生成步 q_len=1 → 可见全部历史）
      - GQA：kv head 数 = q head 数 / num_key_value_groups，手动 repeat
    官方流程从不传 attention_mask（padding_mask 恒为 None），故不走 varlen 路径。
    """
    if padding_mask is not None:
        raise NotImplementedError(
            "kimi eager 回退不支持 padding_mask（官方流程不传 attention_mask）")
    bsz, q_len, n_heads, head_dim = query_states.shape
    q = query_states.transpose(1, 2)   # (bsz, heads, q_len, head_dim)
    k = key_states.transpose(1, 2)
    v = value_states.transpose(1, 2)
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale   # (bsz, heads, q_len, kv_len)
    kv_len = attn.shape[-1]
    offset = kv_len - q_len
    causal = torch.tril(
        torch.ones(q_len, kv_len, dtype=torch.bool, device=attn.device),
        diagonal=offset,
    )
    attn = attn.masked_fill(~causal, float("-inf"))
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(attn.dtype)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2)


def patch_eager_attention(model):
    """把已加载模型的 attention 实现替换为手动 attention（类级 patch，立即生效）。"""
    attn_cls = type(model.model.layers[0].self_attn)
    attn_cls._flash_attention_forward = _eager_flash_attention_forward


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def _load_whisper_encoder(whisper_path, gpu):
    """加载 kimi 自带的 whisper-large-v3 encoder（transformers 4.46.3 兼容路径）。

    该 safetensors 缺 __metadata__ 头，transformers 4.46.3 的 from_pretrained 会崩
    （modeling_utils.py 里 metadata.get 抛 AttributeError），故手动加载。
    与官方 kimia_infer 的 vendored WhisperEncoder 结构一致（conv2 stride=2，
    3000 mel 帧 → 1500 encoder 帧）。
    """
    from transformers import WhisperConfig, WhisperModel
    import safetensors.torch as st

    cfg = WhisperConfig.from_pretrained(whisper_path)
    model = WhisperModel(cfg)
    sd = st.load_file(os.path.join(whisper_path, "model.safetensors"), device="cpu")
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    encoder = model.encoder.to(f"cuda:{gpu}").to(torch.bfloat16).eval()
    encoder.requires_grad_(False)
    del model
    torch.cuda.empty_cache()
    return encoder


def load_kimi(model_path, gpu, freeze=True):
    """加载 Kimi-Audio-7B-Instruct，返回 (model, tokenizer, extra_tokens, whisper_encoder, feature_extractor)。

    - model: MoonshotKimiaForCausalLM（bf16，device_map 到 gpu）
    - tokenizer: TikTokenTokenizer（megatron 风格 API！）
    - extra_tokens: 特殊 token id 字典
    - whisper_encoder: bf16 冻结 encoder（可微链的一部分，输入 mel 的叶子张量在外部）
    - feature_extractor: WhisperFeatureExtractor（非可微 mel 提取）
    """
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor,
    )

    torch.cuda.set_device(gpu)
    patch_flash_attn_requirement()

    # 模型内部 forward 用 torch.cuda.current_device() 搬运张量，set_device 保证一致
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": gpu},
        trust_remote_code=True,
    ).eval()
    if freeze:
        model.requires_grad_(False)
    patch_eager_attention(model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    extra_tokens = _build_extra_tokens(tokenizer)

    whisper_encoder = _load_whisper_encoder(
        os.path.join(model_path, "whisper-large-v3"), gpu)
    fe = WhisperFeatureExtractor.from_pretrained(
        os.path.join(model_path, "whisper-large-v3"))

    print(f"Kimi-Audio loaded on cuda:{gpu} "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params, bf16)")
    return model, tokenizer, extra_tokens, whisper_encoder, fe


def _build_extra_tokens(tokenizer):
    """特殊 token id（用 tokenizer.special_tokens，不用 convert_tokens_to_ids）。"""
    st_ = tokenizer.special_tokens
    return {
        "msg_end": st_["<|im_msg_end|>"],
        "user_msg_start": st_["<|im_user_msg_start|>"],
        "assistant_msg_start": st_["<|im_assistant_msg_start|>"],
        "media_begin": st_["<|im_media_begin|>"],
        "media_end": st_["<|im_media_end|>"],
        "kimia_text_blank": st_["<|im_kimia_text_blank|>"],
        "kimia_text_eos": st_["<|im_kimia_text_eos|>"],
        "kimia_user_msg_start": st_["<|im_kimia_user_msg_start|>"],
        "kimia_assistant_msg_start": st_["<|im_kimia_assistant_msg_start|>"],
        "kimia_speech_ct_id": st_["<|im_kimia_speech_ct_id|>"],
        "kimia_speech_ctd_id": st_["<|im_kimia_speech_ctd_id|>"],
    }


# ---------------------------------------------------------------------------
# 输入构造（对齐官方 kimia_infer/api/prompt_manager.py）
# ---------------------------------------------------------------------------

def token_len_of(n_samples):
    """每 80ms（1280 个 16k 采样点）1 个音频 token，公式取自官方 forward。"""
    return (n_samples - 1) // (160 * 8) + 1


def extract_mel(fe, audio_path, device="cpu", max_chunks=None):
    """waveform → 补零到 30s → log-mel (1,128,3000)（与官方 log_mel_spectrogram 逐位一致）。

    返回 (mel_tensor, token_len)。官方流程把音频按 30s 分块，每块独立提特征后拼接；
    本项目数据集（advbench/libri）均 < 30s，单块即可。超过 30s 的音频在此截断到 30s
    （与官方前 30s 行为一致，如有需要可后续扩展多块拼接）。
    """
    wav, sr = librosa.load(audio_path, sr=16000)
    L = min(wav.shape[0], 30 * 16000)
    wav = wav[:L]
    token_len = token_len_of(L)
    if L < 480000:
        wav = np.pad(wav, (0, 480000 - L))
    mel = fe(wav, sampling_rate=16000, return_tensors="pt", padding=True)["input_features"]
    mel = mel.to(device)
    return mel, token_len


def whisper_features(whisper_encoder, mel, token_len):
    """可微链核心: mel (1,128,3000) → encoder 输出 → 截断 token_len*4 帧 → (1, token_len, 5120)。

    与官方 extract_whisper_feat 一致（encoder 2x 压缩 → 每 4 个 encoder 帧 reshape 成
    一行 5120 维，即每 token 覆盖 80ms）。mel 叶子为 fp32，内部转 bf16 喂 encoder，
    梯度可回传到 mel。
    """
    mel_bf = mel.to(torch.bfloat16)
    enc_out = whisper_encoder(mel_bf, return_dict=True).last_hidden_state  # (1,1500,1280)
    enc_out = enc_out[:, : token_len * 4, :]
    return enc_out.reshape(1, token_len, 5120)


def build_prompt_tokens(extra_tokens, tokenizer, token_len, device,
                        question=None, target_text=None):
    """构造双流 token 序列（对齐官方 get_prompt），返回 (audio_ids, text_ids,
    is_continuous_mask, target_ids)。

    布局（与官方逐 token 一致）:
      [user_msg_start, (question tokens), media_begin, blank×token_len, media_end,
       speech_ct_id, msg_end, assistant_msg_start, (target tokens)]
    - 音频流 span 用 kimia_text_blank 占位（官方此处是 VQ 语音 token；没有 VQ
      tokenizer 时用 blank。whisper 特征范数 ~1.9 远大于 blank embedding 范数 ~0.18，
      音频内容由特征主导，占位选择对行为影响可忽略——已实测验证）
    - question 与音频在同一 user 消息（官方 [user-text, user-audio] 同角色合并后
      与单消息序列完全相同）
    - target_text 追加在 assistant 起始之后（文本流），用于 CE 训练
    """
    audio_ids, text_ids, is_cont = [], [], []

    # user 角色起始
    audio_ids.append(extra_tokens["kimia_user_msg_start"])
    text_ids.append(extra_tokens["kimia_text_blank"])
    is_cont.append(0)

    if question is not None:
        q_ids = tokenizer.encode(question, bos=False, eos=False)
        text_ids += q_ids
        audio_ids += [extra_tokens["kimia_text_blank"]] * len(q_ids)
        is_cont += [0] * len(q_ids)

    # 音频消息: media_begin + span + media_end + ct + msg_end
    audio_ids += [extra_tokens["media_begin"]] \
        + [extra_tokens["kimia_text_blank"]] * token_len \
        + [extra_tokens["media_end"]]
    text_ids += [extra_tokens["kimia_text_blank"]] * (token_len + 2)
    is_cont += [0] + [1] * token_len + [0]

    audio_ids.append(extra_tokens["kimia_speech_ct_id"])  # output_type="text"
    text_ids.append(extra_tokens["kimia_text_blank"])
    is_cont.append(0)
    audio_ids.append(extra_tokens["msg_end"])
    text_ids.append(extra_tokens["kimia_text_blank"])
    is_cont.append(0)

    # assistant 起始
    audio_ids.append(extra_tokens["kimia_assistant_msg_start"])
    text_ids.append(extra_tokens["kimia_text_blank"])
    is_cont.append(0)

    target_ids = None
    if target_text is not None:
        tgt = tokenizer.encode(target_text, bos=False, eos=False)
        target_ids = torch.tensor(tgt, dtype=torch.long, device=device)
        audio_ids += [extra_tokens["kimia_text_blank"]] * len(tgt)
        text_ids += tgt
        is_cont += [0] * len(tgt)

    aids = torch.tensor([audio_ids], dtype=torch.long, device=device)
    tids = torch.tensor([text_ids], dtype=torch.long, device=device)
    imask = torch.tensor([is_cont], dtype=torch.bool, device=device)

    # 与模型内部 assert 一致: span 长 - 1 == is_continuous_mask.sum()
    span = (aids == extra_tokens["media_end"]).nonzero()[0, 1] \
        - (aids == extra_tokens["media_begin"]).nonzero()[0, 1]
    assert span - 1 == int(imask.sum()), (span, int(imask.sum()))
    return aids, tids, imask, target_ids


# ---------------------------------------------------------------------------
# 前向 / 生成
# ---------------------------------------------------------------------------

def kimi_forward(model, audio_ids, text_ids, is_continuous_mask, whisper_feature,
                 position_ids=None, past_key_values=None, use_cache=False):
    """包装 model(...) 双流前向，返回 (audio_logits, text_logits, past_key_values)。

    对齐官方 _generate_loop 的调用方式（return_dict=False；无 attention_mask）。
    whisper_feature / is_continuous_mask 在生成步（past 非空）传 None。
    """
    out = model(
        input_ids=audio_ids,
        text_input_ids=text_ids,
        whisper_input_feature=whisper_feature,
        is_continuous_mask=is_continuous_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        return_dict=False,
    )
    if use_cache:
        return out[0], out[1], out[2]
    return out[0], out[1], None


@torch.no_grad()
def kimi_generate(model, whisper_encoder, extra_tokens, tokenizer, device,
                  mel, audio_ids, text_ids, is_continuous_mask, token_len,
                  question=None, max_new_tokens=1024, verbose=False):
    """官方风格生成循环（text-only, greedy）。

    - prefill: 全序列 + whisper 特征
    - 之后每步: 音频流 blank、文本流 = 上一步采样文本，position_ids = last+1，
      带 past_key_values，whisper_feature/is_continuous_mask=None
    - 停止: 采样到 kimia_text_eos，或达到 max_new_tokens
    - 返回: 文本（token < kimia_token_offset 过滤后 decode，与官方 detokenize_text 一致）
    """
    wf = whisper_features(whisper_encoder, mel, token_len)
    position_ids = torch.arange(0, audio_ids.shape[1], dtype=torch.long,
                                device=device).unsqueeze(0)
    _, text_logits, past = kimi_forward(
        model, audio_ids, text_ids, is_continuous_mask, wf,
        position_ids=position_ids, past_key_values=None, use_cache=True)
    last_pos = audio_ids.shape[1] - 1

    gen_ids = []
    a_in = torch.full((1, 1), extra_tokens["kimia_text_blank"],
                      dtype=torch.long, device=device)
    eos_id = extra_tokens["kimia_text_eos"]
    offset = 152064  # kimia_token_offset

    for i in range(max_new_tokens):
        if i == 0:
            nid = int(torch.argmax(text_logits[0, -1, :]).item())
        else:
            _, text_logits, past = kimi_forward(
                model, a_in, t_in, None, None,
                position_ids=pos_ids, past_key_values=past, use_cache=True)
            nid = int(torch.argmax(text_logits[0, -1, :]).item())
        if nid == eos_id:
            break
        gen_ids.append(nid)
        t_in = torch.tensor([[nid]], dtype=torch.long, device=device)
        pos_ids = torch.tensor([[last_pos + 1]], dtype=torch.long, device=device)
        last_pos += 1
        if verbose:
            print(f"  [{i}] {tokenizer.decode([nid])!r}")

    text = tokenizer.decode([t for t in gen_ids if t < offset])
    return text


# ---------------------------------------------------------------------------
# 可微前向（saliency / 扰动训练用）
# ---------------------------------------------------------------------------

def kimi_jb_logits(model, whisper_encoder, mel, audio_ids, text_ids,
                   is_continuous_mask, token_len):
    """mel 叶子 → 完整可微链 → 文本流 logits (1, S, V)。

    mel 必须 requires_grad=True。模型与 encoder 均冻结（requires_grad=False），
    只有 mel 得到梯度。
    """
    wf = whisper_features(whisper_encoder, mel, token_len)
    audio_logits, text_logits, _ = kimi_forward(
        model, audio_ids, text_ids, is_continuous_mask, wf, use_cache=False)
    return text_logits
