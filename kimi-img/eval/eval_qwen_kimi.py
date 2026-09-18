#!/usr/bin/env python3
"""Run Kimi-Audio inference on a manifest, preserving row order."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
KIMI = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(KIMI)]
from kimi_adapter import (build_prompt_tokens, extract_mel, kimi_generate,  # noqa: E402
                          load_kimi)


def align_time(tensor, length):
    if tensor.shape[-1] >= length:
        return tensor[..., :length]
    return torch.nn.functional.pad(tensor, (0, length - tensor.shape[-1]))


def load_perturb(path):
    return torch.load(path, map_location="cpu")["PTB"]


@torch.no_grad()
def kimi_infer(audio_path, question, model, whisper_encoder, feature_extractor,
               tokenizer, extra_tokens, device, perturb, max_new_tokens=1024):
    mel, token_len = extract_mel(feature_extractor, audio_path, device=device)
    if isinstance(perturb, torch.Tensor):
        delta = align_time(perturb.to(device=device, dtype=torch.bfloat16), mel.shape[-1])
        mel += delta.to(mel.dtype)
    audio_ids, text_ids, continuous_mask, _ = build_prompt_tokens(
        extra_tokens, tokenizer, token_len, device,
        question=question, target_text=None)
    return kimi_generate(
        model, whisper_encoder, extra_tokens, tokenizer, device, mel,
        audio_ids, text_ids, continuous_mask, token_len,
        question=question, max_new_tokens=max_new_tokens)


def read_manifest(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--perturb_path")
    parser.add_argument("--no_perturb", action="store_true")
    parser.add_argument("--model_path", default=ROOT / "models/Kimi-Audio-7B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()
    if not args.no_perturb and not args.perturb_path:
        parser.error("--perturb_path is required unless --no_perturb is set")

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    model, tokenizer, extra_tokens, whisper_encoder, feature_extractor = load_kimi(
        args.model_path, args.gpu)
    device = f"cuda:{args.gpu}"
    perturb = 0.0 if args.no_perturb else load_perturb(args.perturb_path)
    responses = [kimi_infer(
        row["audio"], row.get("prompt"), model, whisper_encoder,
        feature_extractor, tokenizer, extra_tokens, device, perturb,
        args.max_new_tokens) for row in tqdm(rows, desc="Kimi-Audio")]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(responses, stream)
    print(f"saved {len(responses)} responses to {output}")


if __name__ == "__main__":
    main()
