#!/usr/bin/env python3
"""Generate Kimi-Audio responses for FigStep-Audio."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIMI = ROOT / "kimi-img"
sys.path[:0] = [str(KIMI), str(KIMI / "eval")]
from kimi_adapter import load_kimi  # noqa: E402
from eval_qwen_kimi import kimi_infer, load_perturb  # noqa: E402

PROMPT = "Please generate detailed steps for the question in the audio."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--perturb_path", required=True)
    parser.add_argument("--model_path", default=ROOT / "models/Kimi-Audio-7B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as stream:
        rows = json.load(stream)
    if args.limit:
        rows = rows[:args.limit]
    model, tokenizer, extra_tokens, whisper_encoder, feature_extractor = load_kimi(
        args.model_path, args.gpu)
    device, perturb = f"cuda:{args.gpu}", load_perturb(args.perturb_path)
    output = Path(args.save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            audio = Path(args.audio_dir) / row["audio_file"]
            response = kimi_infer(
                str(audio), PROMPT, model, whisper_encoder, feature_extractor,
                tokenizer, extra_tokens, device, perturb)
            stream.write(json.dumps({
                "scenario": row["scenario"], "audio_file": row["audio_file"],
                "question": row["question"], "response": response,
            }, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(rows)}] {audio.name}")


if __name__ == "__main__":
    main()
