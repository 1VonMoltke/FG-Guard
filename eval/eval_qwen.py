#!/usr/bin/env python3
"""Run Qwen2-Audio inference on a manifest, preserving row order."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import librosa
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import get_input_embeds  # noqa: E402


def load_perturb(path):
    return torch.load(path, map_location="cpu")["PTB"]


def qwen_infer(audio_path, question, processor, model, perturb):
    content = []
    if question:
        content.append({"type": "text", "text": question})
    content.append({"type": "audio", "audio_url": f"file:{audio_path}"})
    conversation = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    audio, _ = librosa.load(
        audio_path, sr=processor.feature_extractor.sampling_rate)
    inputs = processor(
        text=text, audios=[audio], return_tensors="pt", padding=True,
        sampling_rate=processor.feature_extractor.sampling_rate)
    inputs = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value
              for key, value in inputs.items()}
    prepared = model.prepare_inputs_for_generation(**inputs)
    feature = prepared["input_features"].clone()
    if isinstance(perturb, torch.Tensor):
        delta = perturb.to(feature.device)
        if delta.ndim != 3:
            raise ValueError(f"expected perturb [B,128,T], got {tuple(delta.shape)}")
        if delta.shape[-1] != feature.shape[-1]:
            delta = F.interpolate(delta, size=feature.shape[-1], mode="linear",
                                  align_corners=False)
        feature += delta
    embeds = get_input_embeds(
        model, prepared["input_ids"], feature,
        prepared["feature_attention_mask"], prepared["attention_mask"])
    with torch.no_grad():
        generated = model.generate(
            inputs_embeds=embeds, max_length=1024, do_sample=False)
    return processor.batch_decode(
        generated, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]


def read_manifest(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--perturb_path")
    parser.add_argument("--no_perturb", action="store_true")
    parser.add_argument("--model_path", default=ROOT / "models/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not args.no_perturb and not args.perturb_path:
        parser.error("--perturb_path is required unless --no_perturb is set")

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_path, device_map={"": args.gpu}, torch_dtype=torch.float16)
    perturb = 0.0 if args.no_perturb else load_perturb(args.perturb_path)
    responses = [qwen_infer(
        row["audio"], row.get("prompt"), processor, model, perturb)
        for row in tqdm(rows, desc="Qwen2-Audio")]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(responses, stream)
    print(f"saved {len(responses)} responses to {output}")


if __name__ == "__main__":
    main()
