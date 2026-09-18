#!/usr/bin/env python3
"""Evaluate Qwen2-Audio transcription utility under a learned perturbation."""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import calc_asr_metrics, extract_single_quoted_text, get_json_pairs  # noqa: E402
from eval_qwen import load_perturb, qwen_infer  # noqa: E402

PROMPT = "Transcribe the audio into text word-for-word, preserving all words accurately."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--perturb_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_path", default=ROOT / "models/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_path, device_map={"": args.gpu}, torch_dtype=torch.float16)
    perturb = load_perturb(args.perturb_path)
    results = []
    for audio, reference in tqdm(get_json_pairs(args.data_path, "wav", "txt"), desc="ASR"):
        response = qwen_infer(audio, PROMPT, processor, model, perturb)
        results.append((reference, extract_single_quoted_text(response)))
    summary = calc_asr_metrics(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
