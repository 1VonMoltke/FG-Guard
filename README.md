# FG-Guard

FG-Guard learns universal defensive perturbations on log-Mel features of audio models. This repository contains the paper protocol: **2D saliency masks, fixed 35% coverage, and a mixed refusal-CE + ASR-CE loss**. It supports Qwen2-Audio-7B-Instruct and Kimi-Audio-7B-Instruct.

The repository does not include 1D masks, Gaussian noise, Local Smooth, or GPU preemption/queueing code.

## Method and default settings

Saliency is the ratio between the gradient of the harmful-request refusal target and the Whisper ASR gradient. The top 35% of positions are selected on the full `(128, T)` time-frequency plane:

```text
L = CE(model(audio_harmful + delta), refusal)
  + lambda * CE(model(audio_safe + delta), transcript)
```

Defaults are `mask_ratio=0.35`, `lambda=1`, `tau=0.5`, `lr=3e-4`, `iterations=2000`, and `seed=42`. Both model frontends use audio windows of at most 30 seconds; longer audio is truncated by the frontend and is not pre-filtered.

## Repository layout

```text
compute_saliency_2d.py       Qwen 2D saliency computation
train/main_2d_mixed.py       Qwen mixed-loss training
eval/                        SRoA, ASR, AIR-Bench, and FigStep-Audio evaluation
kimi-img/                    Kimi implementation
scripts/run_{qwen,kimi}35.sh Complete single-dataset pipelines
tools/prepare_data.py        Deterministic data split with seed=42
data/advwave/                39 AdvWave clips distributed with the repository
AIR-bench/                   AIR-Bench assembly and scoring scripts
whisper/                     MIT-licensed OpenAI Whisper implementation
```

## Installation

Python 3.10 and CUDA 12.1 are recommended. Make sure `ffmpeg` is installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Adjust the PyTorch/CUDA combination to match your driver. The code was validated with `torch==2.2.2` and `transformers==4.46.3`.

## Download model weights

```bash
mkdir -p models
hf download Qwen/Qwen2-Audio-7B-Instruct \
  --local-dir models/Qwen2-Audio-7B-Instruct
hf download moonshotai/Kimi-Audio-7B-Instruct \
  --local-dir models/Kimi-Audio-7B-Instruct
hf download sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406 \
  --local-dir models/sorrybench
wget -O models/large-v3.pt \
  https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt
```

SorryBench is gated. Accept its Hugging Face license first, then run `hf auth login`.

## Download datasets

Except for `data/advwave/`, datasets are not distributed with this repository. Keep them under `datasets/`:

```bash
mkdir -p datasets
hf download WeifeiJin/AdvBench-Audio --repo-type dataset \
  --local-dir datasets/AdvBench-Audio
hf download MBZUAI/AudioJailbreak --repo-type dataset \
  --local-dir datasets/AudioJailbreak
hf download researchtopic/Jailbreak-AudioBench --repo-type dataset \
  --local-dir datasets/Jailbreak-AudioBench
git clone https://github.com/Researchtopic/Code-Jailbreak-AudioBench.git datasets/Code-Jailbreak-AudioBench
git clone https://github.com/OFA-Sys/AIR-Bench.git datasets/AIR-Bench-code
hf download qyang1021/AIR-Bench-Dataset --repo-type dataset \
  --local-dir datasets/AIR-Bench
git clone https://github.com/linweiii/SARSteer.git datasets/SARSteer
```

The full Jailbreak-AudioBench release is large. The 3,725-clip subset used here contains `original=520`, `accent=1560`, `emotion=1040`, and `emphasis=605`. Align its parquet audio with `Text/Explicit_Advbench.csv` from `Code-Jailbreak-AudioBench` by `audio_NNN`:

```text
datasets/Jailbreak-AudioBench/
├── original/audio_NNN.wav
├── accent/<variant>/audio_NNN.wav
├── emotion/<variant>/audio_NNN.wav
├── emphasis/<variant>/audio_NNN.wav
└── jailbreakaudiobench_pairs.jsonl  # {"prompt": ..., "audio": "relative/path"}
```

AudioJailbreak uses `audio/AJailbreak_Base/*.wav`, `convert/question/wav_combined_output.jsonl`, and `audio/jailbreak_llms/jailbreak_llms_forbidden.jsonl`. If `almguard_view/.../rename_map.jsonl` exists, the preparation script uses it; otherwise it reads the base audio directory.

FigStep-Audio uses the harmful/safe audio produced by the SARSteer release procedure (350 clips per split); this protocol uses fixed 250-clip test lists. Follow all upstream licenses: AdvBench-Audio is CC BY-NC 4.0, AudioJailbreak is Apache-2.0, and Jailbreak-AudioBench code is MIT-licensed.

## Data splits

Prepare a safe ASR JSONL manifest with at least 280 valid audio files. Each row must contain `wav`/`txt` (or `audio`/`reference`):

```json
{"wav":"/absolute/path/to/audio.wav","txt":"reference transcript","dataset":"LibriSpeech"}
```

The paper protocol selects 280 entries from local LibriSpeech and AIR-Bench transcripts and splits them with seed 42 into `168/56/56`:

```bash
python3 tools/prepare_data.py \
  --datasets_root datasets \
  --safe_manifest datasets/safe_asr.jsonl \
  --output data/processed
```

| Dataset | Harmful train | Validation | Test | Safe train/test |
|---|---:|---:|---:|---:|
| AdvBench-Audio | 416 | 52 | 52 | 168 / 56 |
| AudioJailbreak | 1212 | 129 | 152 | 168 / 56 |
| Jailbreak-AudioBench | 2980 | 374 | 371 | 168 / 56 |
| AdvWave fold 0 | 20 | 0 | 19 | 168 / 56 |
| AdvWave fold 1 | 19 | 0 | 20 | 168 / 56 |

AdvBench and Jailbreak-AudioBench are split by the 520 original harmful goals into 80/10/10 partitions, so variants of the same goal never cross splits. AudioJailbreak is grouped by normalized harmful goal and stratified by source; its 18 goals overlapping with AdvBench remain in the test set. AdvWave uses two disjoint group-based folds covering all 39 clips. `train_harmful_wavs/` contains symlinks and does not duplicate audio.

Each trainable directory contains:

```text
data/processed/<dataset>/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── test_safe_asr.jsonl
└── train_harmful_wavs/
```

AdvWave is stored under `data/processed/advwave/fold0` and `fold1`.

## Complete training and evaluation

The scripts run saliency computation, training, harmful-set inference, SorryBench grouped judging, and safe ASR evaluation. The GPU argument can be one card or `model_gpu,whisper/judge_gpu`:

```bash
bash scripts/run_qwen35.sh 2,3 \
  data/processed/advbench outputs/qwen/advbench

bash scripts/run_kimi35.sh 2,3 \
  data/processed/advbench outputs/kimi/advbench
```

Replace the dataset directory with `audiojailbreak`, `jailbreak_audiobench`, `advwave/fold0`, or `advwave/fold1`. Each output contains `mask_2d_35.npz`, the final perturbation, `responses.pkl`, `sroa.json`, per-example judgments, and `asr.json`.

## AIR-Bench transfer evaluation

For Qwen, run the following commands; for Kimi, replace the evaluator with `eval_airbench_kimi.py`:

```bash
python3 eval/eval_airbench_qwen.py \
  --gpu 2 --perturb_path outputs/qwen/advbench/train/perturb_2d_mixed.pth \
  --meta datasets/AIR-Bench/Chat/Chat_meta.json \
  --audio_dir datasets/AIR-Bench/Chat/speech_QA_common_voice_en \
  --save_path outputs/air/raw.jsonl

python3 AIR-bench/assemble.py outputs/air/raw.jsonl \
  datasets/AIR-Bench/Chat/Chat_meta.json outputs/air/Chat_result.jsonl

DEEPSEEK_API_KEY=... DEEPSEEK_BASE_URL=https://api.deepseek.com \
python3 AIR-bench/score_chat.py -r outputs/air -mr Chat_result.jsonl \
  -i judge_input.jsonl -o judge_output.jsonl
python3 AIR-bench/summarize.py outputs/air/judge_output.jsonl
```

The AIR-Bench judge accepts any OpenAI-compatible endpoint; credentials are read only from environment variables.

## FigStep-Audio transfer evaluation

Run inference separately on the 250 harmful and 250 safe test lists, then compute BRR. For Kimi, use `eval_figstep_kimi.py`:

```bash
python3 eval/eval_figstep_qwen.py --gpu 2 \
  --perturb_path outputs/qwen/advbench/train/perturb_2d_mixed.pth \
  --json datasets/FigStep-Audio/figstep_audio/test_250.json \
  --audio_dir datasets/FigStep-Audio/figstep_audio/question \
  --save_path outputs/figstep/harmful.jsonl

python3 eval/eval_figstep_qwen.py --gpu 2 \
  --perturb_path outputs/qwen/advbench/train/perturb_2d_mixed.pth \
  --json datasets/FigStep-Audio/figstep_audio_safe/test_250.json \
  --audio_dir datasets/FigStep-Audio/figstep_audio_safe/question \
  --save_path outputs/figstep/safe.jsonl

python3 eval/judge_figstep.py outputs/figstep/harmful.jsonl \
  outputs/figstep/safe.jsonl --gpu 3
```

SARSteer directory names may vary by release; follow the `audio_file` entries in its published manifests.

## Metrics

- `SRoA ↓`: proportion of harmful requests judged by SorryBench as successfully followed; also reported by dataset/source/category/variant.
- `WER/CER ↓`: safe-speech transcription error rates.
- `AIR Score / Win Rate ↑`: average judge score and win rate against reference answers on AIR-Bench.
- `harmful refusal ↑`: refusal rate on the FigStep harmful set.
- `FRR ↓`: false refusal rate on the FigStep safe set.
- `BRR ↑`: `0.5 × (harmful refusal + safe compliance)`.

This code and data are intended only for authorized safety research and defensive evaluation. Do not use them to generate or deploy harmful content.
