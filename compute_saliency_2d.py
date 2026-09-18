#!/usr/bin/env python3
"""Compute the Qwen2-Audio 2D saliency mask at fixed 35% coverage."""

import argparse
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch
import whisper
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from utils import (compute_saliency_sets_2d, get_audio_file_list,
                   prepare_inputs_and_targets)

ROOT = Path(__file__).resolve().parent
REFUSAL = "I'm sorry, but I cannot fulfill this request."
MASK_RATIO = 0.35


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dirs", nargs="+", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--model_path", default=ROOT / "models/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--asr_path", default=ROOT / "models/large-v3.pt")
    parser.add_argument("--gpu", default="0", help="Qwen GPU[,Whisper GPU]")
    parser.add_argument("--time_block_size", type=int)
    parser.add_argument("--max_audios", type=int)
    args = parser.parse_args()

    gpu_ids = [int(value.strip()) for value in args.gpu.split(",")]
    qwen_gpu, asr_gpu = gpu_ids[0], gpu_ids[-1]
    lists = [get_audio_file_list(directory) for directory in args.wav_dirs]
    audio_files = [item for group in zip_longest(*lists) for item in group if item]
    if args.max_audios:
        audio_files = audio_files[:args.max_audios]
    if not audio_files:
        raise ValueError("no WAV files found")

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_path, device_map={"": qwen_gpu}, torch_dtype=torch.float16)
    asr_model = whisper.load_model(str(args.asr_path), device=f"cuda:{asr_gpu}")
    refusal_maps, asr_maps = [], []
    for index, audio_path in enumerate(audio_files, 1):
        print(f"[{index}/{len(audio_files)}] {audio_path}")
        inputs, target_ids = prepare_inputs_and_targets(
            audio_path, processor, model, REFUSAL)
        refusal, asr = compute_saliency_sets_2d(
            model, asr_model, inputs, target_ids, args.time_block_size)
        refusal_maps.append(refusal)
        asr_maps.append(asr)

    min_time = min(item.shape[1] for item in refusal_maps + asr_maps)
    refusal_stack = np.stack([item[:, :min_time] for item in refusal_maps])
    asr_stack = np.stack([item[:, :min_time] for item in asr_maps])
    avg_refusal, avg_asr = refusal_stack.mean(0), asr_stack.mean(0)
    score = avg_refusal / (avg_asr + 1e-6)
    selected = max(1, int(score.size * MASK_RATIO))
    indices = np.argsort(score, axis=None)[::-1][:selected]
    mask = np.zeros_like(score, dtype=np.float32)
    np.put(mask, indices, 1.0)

    output = Path(args.save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, grad_j_2d=avg_refusal, grad_a_2d=avg_asr,
             score_2d=score, mask_2d=mask,
             var_j_2d=refusal_stack.var(0), var_a_2d=asr_stack.var(0),
             mask_ratio=MASK_RATIO)
    print(f"saved {output}: {int(mask.sum())}/{mask.size} positions (35%)")


if __name__ == "__main__":
    main()
