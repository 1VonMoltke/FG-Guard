#!/usr/bin/env python3
"""Judge manifest-aligned responses and report SRoA by manifest metadata."""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from llm_evaluation import get_sorrybench_pipeline, sorrybench_evaluate  # noqa: E402


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(scores):
    return {"n": len(scores), "sroa": sum(scores) / len(scores) if scores else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model", type=Path, default=ROOT / "models/sorrybench")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with args.responses.open("rb") as f:
        responses = pickle.load(f)
    rows = read_jsonl(args.manifest)
    if len(responses) != len(rows):
        raise ValueError(f"responses={len(responses)} != manifest={len(rows)}")
    if args.limit:
        responses, rows = responses[:args.limit], rows[:args.limit]

    pipe = get_sorrybench_pipeline(args.gpu, str(args.model))
    scores, grouped, records = [], defaultdict(lambda: defaultdict(list)), []
    dimensions = ("dataset", "source", "category", "variant")
    for index, (response, row) in enumerate(tqdm(zip(responses, rows), total=len(rows))):
        gt = row.get("gt") or row.get("harmful_prompt")
        if not gt:
            raise ValueError(f"missing gt at manifest row {index}")
        score, raw = sorrybench_evaluate(gt, response, pipe)
        score = float(score)
        scores.append(score)
        for dimension in dimensions:
            if row.get(dimension) is not None:
                grouped[dimension][str(row[dimension])].append(score)
        records.append({
            "id": row.get("id"), "audio": row["audio"], "score": score,
            "judge_output": raw, **{key: row.get(key) for key in dimensions},
        })

    groups = {}
    for dimension, values in grouped.items():
        entries = {name: summarize(group_scores) for name, group_scores in sorted(values.items())}
        groups[dimension] = {
            "macro_sroa": sum(item["sroa"] for item in entries.values()) / len(entries),
            "values": entries,
        }
    result = {"overall": summarize(scores), "groups": groups}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.records:
        args.records.parent.mkdir(parents=True, exist_ok=True)
        with args.records.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
