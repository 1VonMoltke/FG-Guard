#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 6 )); then
  echo "usage: $0 KIMI_GPU[,WHISPER_GPU] DATASET_DIR RUN_DIR [KIMI_MODEL] [WHISPER_PT] [JUDGE_MODEL]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU_PAIR="$1"
PRIMARY="${GPU_PAIR%%,*}"
JUDGE_GPU="${GPU_PAIR##*,}"
DATA="$2"
RUN="$3"
MODEL="${4:-$ROOT/models/Kimi-Audio-7B-Instruct}"
WHISPER="${5:-$ROOT/models/large-v3.pt}"
JUDGE="${6:-$ROOT/models/sorrybench}"
MASK="$RUN/mask_2d_35.npz"
PERTURB="$RUN/train/perturb_2d_mixed.pth"
RESPONSES="$RUN/responses.pkl"

mkdir -p "$RUN"
python3 "$ROOT/kimi-img/compute_saliency_2d_kimi.py" \
  --gpu "$GPU_PAIR" --model_path "$MODEL" --asr_path "$WHISPER" \
  --wav_dirs "$DATA/train_harmful_wavs" --save_path "$MASK"
python3 "$ROOT/kimi-img/train/main_2d_mixed_kimi.py" \
  --gpu "$PRIMARY" --model_path "$MODEL" --train_jsonl "$DATA/train.jsonl" \
  --mask_path "$MASK" --save_path "$RUN/train" --num_iters 2000 \
  --lam 1 --tau .5 --lr 3e-4
python3 "$ROOT/kimi-img/eval/eval_qwen_kimi.py" \
  --gpu "$PRIMARY" --model_path "$MODEL" --manifest "$DATA/test.jsonl" \
  --perturb_path "$PERTURB" --output "$RESPONSES"
python3 "$ROOT/eval/grouped_judge.py" \
  --gpu "$JUDGE_GPU" --model "$JUDGE" --responses "$RESPONSES" \
  --manifest "$DATA/test.jsonl" --summary "$RUN/sroa.json" \
  --records "$RUN/judged.jsonl"
python3 "$ROOT/kimi-img/eval/eval_asr_kimi.py" \
  --gpu "$PRIMARY" --model_path "$MODEL" --data_path "$DATA/test_safe_asr.jsonl" \
  --perturb_path "$PERTURB" --output "$RUN/asr.json"
