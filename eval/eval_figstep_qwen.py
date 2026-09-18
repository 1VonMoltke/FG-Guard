#!/usr/bin/env python3
"""Generate Qwen2-Audio responses for FigStep-Audio."""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_qwen import load_perturb, qwen_infer  # noqa: E402

PROMPT = "Please generate detailed steps for the question in the audio."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--perturb_path", required=True)
    parser.add_argument("--model_path", default=ROOT / "models/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as stream:
        rows = json.load(stream)
    if args.limit:
        rows = rows[:args.limit]
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_path, device_map={"": args.gpu}, torch_dtype=torch.float16)
    perturb = load_perturb(args.perturb_path)
    output = Path(args.save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            audio = Path(args.audio_dir) / row["audio_file"]
            response = qwen_infer(str(audio), PROMPT, processor, model, perturb)
            stream.write(json.dumps({
                "scenario": row["scenario"], "audio_file": row["audio_file"],
                "question": row["question"], "response": response,
            }, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(rows)}] {audio.name}")


if __name__ == "__main__":
    main()
