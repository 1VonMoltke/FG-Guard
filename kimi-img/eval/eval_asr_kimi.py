#!/usr/bin/env python3
"""Evaluate Kimi-Audio transcription utility under a learned perturbation."""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
KIMI = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(KIMI), str(Path(__file__).resolve().parent)]
from utils import calc_asr_metrics, extract_single_quoted_text, get_json_pairs  # noqa: E402
from kimi_adapter import load_kimi  # noqa: E402
from eval_qwen_kimi import kimi_infer, load_perturb  # noqa: E402

PROMPT = "Transcribe the audio into text word-for-word, preserving all words accurately."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--perturb_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_path", default=ROOT / "models/Kimi-Audio-7B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    model, tokenizer, extra_tokens, whisper_encoder, feature_extractor = load_kimi(
        args.model_path, args.gpu)
    device, perturb = f"cuda:{args.gpu}", load_perturb(args.perturb_path)
    results = []
    for audio, reference in tqdm(get_json_pairs(args.data_path, "wav", "txt"), desc="ASR"):
        response = kimi_infer(
            audio, PROMPT, model, whisper_encoder, feature_extractor,
            tokenizer, extra_tokens, device, perturb)
        results.append((reference, extract_single_quoted_text(response)))
    summary = calc_asr_metrics(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
