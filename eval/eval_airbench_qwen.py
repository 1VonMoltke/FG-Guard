#!/usr/bin/env python3
"""Generate Qwen2-Audio responses for AIR-Bench Chat."""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_qwen import load_perturb, qwen_infer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--perturb_path", required=True)
    parser.add_argument("--dataset", default="common_voice_en")
    parser.add_argument("--model_path", default=ROOT / "models/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.meta, encoding="utf-8") as stream:
        rows = json.load(stream)
    rows = [row for row in rows if not args.dataset or row.get("dataset_name") == args.dataset]
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
            audio = Path(args.audio_dir) / row["path"]
            response = qwen_infer(str(audio), row["question"], processor, model, perturb)
            record = {key: row[key] for key in ("uniq_id", "path", "question", "answer_gt")}
            record["response"] = response
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(rows)}] {audio.name}")
    print(f"saved {len(rows)} responses to {output}")


if __name__ == "__main__":
    main()
