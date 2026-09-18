#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 6 )); then
  echo "usage: $0 QWEN_GPU[,WHISPER_GPU] DATASET_DIR RUN_DIR [QWEN_MODEL] [WHISPER_PT] [JUDGE_MODEL]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU_PAIR="$1"
PRIMARY="${GPU_PAIR%%,*}"
JUDGE_GPU="${GPU_PAIR##*,}"
DATA="$2"
RUN="$3"
MODEL="${4:-$ROOT/models/Qwen2-Audio-7B-Instruct}"
WHISPER="${5:-$ROOT/models/large-v3.pt}"
JUDGE="${6:-$ROOT/models/sorrybench}"
MASK="$RUN/mask_2d_35.npz"
PERTURB="$RUN/train/perturb_2d_mixed.pth"
RESPONSES="$RUN/responses.pkl"

mkdir -p "$RUN"
python3 "$ROOT/compute_saliency_2d.py" \
  --gpu "$GPU_PAIR" --model_path "$MODEL" --asr_path "$WHISPER" \
  --wav_dirs "$DATA/train_harmful_wavs" --save_path "$MASK"
python3 "$ROOT/train/main_2d_mixed.py" \
  --gpu "$PRIMARY" --model_path "$MODEL" --train_jsonl "$DATA/train.jsonl" \
  --mask_path "$MASK" --save_path "$RUN/train" --num_iters 2000 \
  --lam 1 --tau .5 --lr 3e-4 --early_stop_refuse 0
python3 "$ROOT/eval/eval_qwen.py" \
  --gpu "$PRIMARY" --model_path "$MODEL" --manifest "$DATA/test.jsonl" \
  --perturb_path "$PERTURB" --output "$RESPONSES"
python3 "$ROOT/eval/grouped_judge.py" \
  --gpu "$JUDGE_GPU" --model "$JUDGE" --responses "$RESPONSES" \
  --manifest "$DATA/test.jsonl" --summary "$RUN/sroa.json" \
  --records "$RUN/judged.jsonl"
python3 "$ROOT/eval/eval_asr.py" \
  --gpu "$PRIMARY" --model_path "$MODEL" --data_path "$DATA/test_safe_asr.jsonl" \
  --perturb_path "$PERTURB" --output "$RUN/asr.json"
