# FG-Guard

FG-Guard 在音频模型的 log-Mel 特征上学习通用防御扰动。本仓库仅保留论文当前使用的协议：**二维显著性掩码、固定 35% 覆盖率、拒绝 CE + ASR CE 混合损失**。支持 Qwen2-Audio-7B-Instruct 与 Kimi-Audio-7B-Instruct。

本仓库不包含一维掩码、Gaussian Noise、Local Smooth 或 GPU 抢占/排队代码。

## 方法与默认参数

显著性为有害请求拒绝目标的梯度与 Whisper ASR 梯度之比，在完整 `(128, T)` 时频平面上选择前 35% 位置。训练目标为：

```text
L = CE(model(audio_harmful + delta), refusal)
  + lambda * CE(model(audio_safe + delta), transcript)
```

默认参数：`mask_ratio=0.35`、`lambda=1`、`tau=0.5`、`lr=3e-4`、`iterations=2000`、`seed=42`。Qwen2-Audio 和 Kimi-Audio 的前端均使用最多 30 秒的音频窗口；超过 30 秒的内容会被前端截断，本协议不预先过滤长音频。

## 目录

```text
compute_saliency_2d.py       Qwen 二维显著性
train/main_2d_mixed.py       Qwen 混合损失训练
eval/                        SRoA、ASR、AIR-Bench、FigStep-Audio 评测
kimi-img/                    Kimi 对应实现
scripts/run_{qwen,kimi}35.sh 单数据集完整链路
tools/prepare_data.py        固定 seed=42 的确定性数据划分
data/advwave/                随仓库发布的 39 条 AdvWave 音频
AIR-bench/                   AIR-Bench 官方口径组装与打分脚本
whisper/                     显著性计算使用的 OpenAI Whisper 实现（MIT）
```

## 安装

建议 Python 3.10、CUDA 12.1，并确保系统已安装 `ffmpeg`：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch/CUDA 组合应按机器驱动调整；代码验证使用 `torch==2.2.2` 与 `transformers==4.46.3`。

## 模型下载

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

SorryBench 权重是 gated model，需要先在 Hugging Face 接受许可并执行 `hf auth login`。

## 数据下载

除 `data/advwave/` 外，数据集不随仓库分发。建议统一放到 `datasets/`：

```bash
mkdir -p datasets
hf download WeifeiJin/AdvBench-Audio --repo-type dataset \
  --local-dir datasets/AdvBench-Audio
hf download MBZUAI/AudioJailbreak --repo-type dataset \
  --local-dir datasets/AudioJailbreak
hf download researchtopic/Jailbreak-AudioBench --repo-type dataset \
  --local-dir datasets/Jailbreak-AudioBench
git clone https://github.com/Researchtopic/Code-Jailbreak-AudioBench.git \
  datasets/Code-Jailbreak-AudioBench
git clone https://github.com/OFA-Sys/AIR-Bench.git datasets/AIR-Bench-code
hf download qyang1021/AIR-Bench-Dataset --repo-type dataset \
  --local-dir datasets/AIR-Bench
git clone https://github.com/linweiii/SARSteer.git datasets/SARSteer
```

Jailbreak-AudioBench 的完整发布很大。本论文的 3725 条可用子集包含 `original=520`、`accent=1560`、`emotion=1040`、`emphasis=605`；用上游 `Code-Jailbreak-AudioBench` 的 `Text/Explicit_Advbench.csv` 对下载的 parquet 音频按 `audio_NNN` 对齐，整理为：

```text
datasets/Jailbreak-AudioBench/
├── original/audio_NNN.wav
├── accent/<variant>/audio_NNN.wav
├── emotion/<variant>/audio_NNN.wav
├── emphasis/<variant>/audio_NNN.wav
└── jailbreakaudiobench_pairs.jsonl  # {"prompt": ..., "audio": "相对路径"}
```

AudioJailbreak 使用下载包中的 `audio/AJailbreak_Base/*.wav`、`convert/question/wav_combined_output.jsonl` 与 `audio/jailbreak_llms/jailbreak_llms_forbidden.jsonl`。如包内有 `almguard_view/.../rename_map.jsonl`，准备脚本会使用它；否则直接读取基础音频目录。

FigStep-Audio 使用 SARSteer 发布流程产生的 harmful/safe 各 350 条音频，本实验分别取其固定 250 条测试清单。请遵守各上游数据集的许可；AdvBench-Audio 为 CC BY-NC 4.0，AudioJailbreak 为 Apache-2.0，Jailbreak-AudioBench 代码为 MIT，数据中的原始提示还受其来源数据许可约束。

## 数据划分

准备一个安全 ASR JSONL，至少 280 条有效音频。每行使用 `wav/txt`（或 `audio/reference`）字段：

```json
{"wav":"/absolute/path/to/audio.wav","txt":"reference transcript","dataset":"LibriSpeech"}
```

论文实验从本地可用的 LibriSpeech 与 AIR-Bench 转录中固定选择 280 条，使用 seed 42 按 `168/56/56` 划分：

```bash
python3 tools/prepare_data.py \
  --datasets_root datasets \
  --safe_manifest datasets/safe_asr.jsonl \
  --output data/processed
```

输出协议如下：

| 数据集 | harmful train | validation | test | safe train/test |
|---|---:|---:|---:|---:|
| AdvBench-Audio | 416 | 52 | 52 | 168 / 56 |
| AudioJailbreak | 1212 | 129 | 152 | 168 / 56 |
| Jailbreak-AudioBench | 2980 | 374 | 371 | 168 / 56 |
| AdvWave fold 0 | 20 | 0 | 19 | 168 / 56 |
| AdvWave fold 1 | 19 | 0 | 20 | 168 / 56 |

AdvBench/Jailbreak-AudioBench 按 520 个原始 harmful goal 分组为 80/10/10，保证同一 goal 的音频变体不会跨集合。AudioJailbreak 按规范化 harmful goal 分组并按来源分层；与 AdvBench 重合的 18 条保留在测试集。AdvWave 按编号 group 做两折，两个测试折无重叠并覆盖全部 39 条。`train_harmful_wavs/` 是指向原音频的软链接，不会复制大型数据。

每个可训练目录统一为：

```text
data/processed/<dataset>/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── test_safe_asr.jsonl
└── train_harmful_wavs/
```

AdvWave 位于 `data/processed/advwave/fold0` 与 `fold1`。

## 完整训练与测评

脚本顺序执行掩码、训练、有害集推理、SorryBench 分组判定和安全 ASR 测评。GPU 参数可给一张卡，或给 `模型卡,Whisper/判官卡`：

```bash
bash scripts/run_qwen35.sh 2,3 \
  data/processed/advbench outputs/qwen/advbench

bash scripts/run_kimi35.sh 2,3 \
  data/processed/advbench outputs/kimi/advbench
```

将数据目录依次替换为 `audiojailbreak`、`jailbreak_audiobench`、`advwave/fold0`、`advwave/fold1` 即可完成全部实验。每个输出目录包含 `mask_2d_35.npz`、最终扰动、`responses.pkl`、`sroa.json`、逐条判定和 `asr.json`。

## AIR-Bench 迁移评测

以 Qwen 为例，Kimi 将脚本名替换为 `eval_airbench_kimi.py`：

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

AIR-Bench 的 API 判官接受任意 OpenAI-compatible endpoint；密钥只从环境变量读取。

## FigStep-Audio 迁移评测

对 harmful 与 safe 的 250 条清单分别推理，再计算 BRR。Kimi 使用 `eval_figstep_kimi.py`：

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

SARSteer 的具体目录名可能随版本变化，以其发布清单中的 `audio_file` 为准。

## 指标

- `SRoA ↓`：SorryBench 判断模型服从有害请求的比例；同时按 dataset/source/category/variant 分组。
- `WER/CER ↓`：安全语音转写性能损失。
- `AIR Score / Win Rate ↑`：AIR-Bench 回答相对参考答案的平均判官分与胜率。
- `harmful refusal ↑`：FigStep harmful 集拒绝率。
- `FRR ↓`：FigStep safe 集误拒率。
- `BRR ↑`：`0.5 × (harmful refusal + safe compliance)`。

本代码和数据仅用于经授权的安全研究与防御评估，请勿用于生成或部署有害内容。
