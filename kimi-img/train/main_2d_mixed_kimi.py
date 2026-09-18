#!/usr/bin/env python3
"""Train a Kimi-Audio 2D perturbation with refusal + ASR CE losses."""

import argparse
import datetime
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KIMI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, KIMI):
    if path not in sys.path:
        sys.path.insert(0, path)

from kimi_adapter import kimi_jb_logits, load_kimi  # noqa: E402
from utils_kimi import prepare_inputs_and_targets_kimi  # noqa: E402


REFUSAL_TARGET = "I'm sorry, but I cannot fulfill this request."
ASR_PROMPT = "Transcribe the audio into text word-for-word, preserving all words accurately."


def load_dataset(path):
    harmful, benign = [], []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("is_harmful"):
                harmful.append(row)
            elif row.get("task") == "asr":
                benign.append(row)
    print(f"[data] harmful={len(harmful)} benign_asr={len(benign)} ({path})")
    if not harmful or not benign:
        raise ValueError("training JSONL must contain harmful and benign ASR rows")
    return harmful, benign


def load_mask(path, device):
    mask = torch.from_numpy(np.load(path)["mask_2d"]).float()[None]
    if mask.shape[-1] != 3000:
        mask = F.interpolate(mask[None], size=(128, 3000), mode="bilinear",
                             align_corners=False)[0]
    mask = (mask > .5).float().to(device)
    print(f"[mask] {path}: shape={tuple(mask.shape)} nonzero={int(mask.sum())}/{mask.numel()}")
    return mask


def loss_gradient(row, target, question, perturb, model, whisper_encoder, fe,
                  tokenizer, extra_tokens, device):
    mel, aids, tids, imask, token_len, target_ids = prepare_inputs_and_targets_kimi(
        row["audio"], fe, tokenizer, extra_tokens, device,
        target_text=target, question=question)
    logits = kimi_jb_logits(model, whisper_encoder, mel + perturb, aids, tids,
                            imask, token_len)
    shift = logits.size(1) - target_ids.size(0)
    target_logits = logits[..., shift - 1:-1, :].contiguous()
    loss = F.cross_entropy(target_logits.view(-1, target_logits.size(-1)),
                           target_ids.view(-1))
    gradient = torch.autograd.grad(loss, perturb)[0]
    value = loss.item()
    if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
        raise FloatingPointError(row.get("id", row["audio"]))
    return value, gradient


def train(args, harmful, benign, model, whisper_encoder, fe, tokenizer,
          extra_tokens, device):
    mask = load_mask(args.mask_path, device)
    torch.manual_seed(args.seed)
    perturb = (0.01 * torch.randn_like(mask) * mask).requires_grad_(True)
    optimizer = torch.optim.Adam([perturb], lr=args.lr)
    rng = random.Random(args.seed)
    start = time.time()

    for iteration in range(1, args.num_iters + 1):
        h, b = rng.choice(harmful), rng.choice(benign)
        try:
            refusal, grad_h = loss_gradient(
                h, REFUSAL_TARGET, None, perturb, model, whisper_encoder, fe,
                tokenizer, extra_tokens, device)
            asr, grad_b = loss_gradient(
                b, b["reference"], b.get("prompt") or ASR_PROMPT, perturb,
                model, whisper_encoder, fe, tokenizer, extra_tokens, device)
        except FloatingPointError as error:
            print(f"[skip nonfinite] it={iteration} sample={error}")
            torch.cuda.empty_cache()
            continue

        perturb.grad = ((grad_h + args.lam * grad_b) * mask).float()
        optimizer.step()
        with torch.no_grad():
            perturb.clamp_(-args.tau, args.tau).mul_(mask)
            if not torch.isfinite(perturb).all():
                raise FloatingPointError(f"perturb became non-finite at {iteration}")
        optimizer.zero_grad()

        if iteration == 1 or iteration % args.log_every == 0:
            elapsed = datetime.timedelta(seconds=int(time.time() - start))
            print(f"[it {iteration:>5}/{args.num_iters}] {elapsed} | "
                  f"refuse={refusal:.4f} asr={asr:.4f} "
                  f"|delta|_1={float(perturb.detach().abs().sum()):.3f}")
        if iteration % args.save_every == 0:
            torch.save({"PTB": perturb.detach().cpu()}, os.path.join(
                args.save_path, f"perturb_2d_mixed_step{iteration}.pth"))
        torch.cuda.empty_cache()

    output = os.path.join(args.save_path, "perturb_2d_mixed.pth")
    torch.save({"PTB": perturb.detach().cpu()}, output)
    print(f"[done] {output} shape={tuple(perturb.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--model_path", default=os.path.join(
        ROOT, "models", "Kimi-Audio-7B-Instruct"))
    parser.add_argument("--num_iters", type=int, default=2000)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    harmful, benign = load_dataset(args.train_jsonl)
    if args.check_only:
        mask = np.load(args.mask_path)["mask_2d"]
        if mask.shape[0] != 128 or not np.isfinite(mask).all():
            raise ValueError(f"invalid mask shape/data: {mask.shape}")
        print(f"[check] dataset and mask valid: {mask.shape}")
        return
    model, tokenizer, extra_tokens, whisper_encoder, fe = load_kimi(
        args.model_path, args.gpu)
    train(args, harmful, benign, model, whisper_encoder, fe, tokenizer,
          extra_tokens, f"cuda:{args.gpu}")


if __name__ == "__main__":
    main()
