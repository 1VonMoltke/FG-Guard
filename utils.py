"""Shared helpers for 2D saliency, inference, and ASR evaluation."""

import json
import os
import re

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import whisper


def prepare_inputs_and_targets(audio_path, processor, model, target_text):
    audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio_url": f"file:{audio_path}"},
    ]}]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    target_ids = processor(
        text=target_text, return_tensors="pt", padding=True)["input_ids"].to(model.device)
    inputs = processor(
        text=prompt + target_text, audios=[audio], return_tensors="pt",
        padding=True, sampling_rate=sr)
    inputs = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value
              for key, value in inputs.items()}
    prepared = model.prepare_inputs_for_generation(**inputs)
    return {
        key: prepared[key].to(model.device)
        for key in ("input_ids", "attention_mask", "input_features", "feature_attention_mask")
    }, target_ids


def get_input_embeds(model, input_ids, input_features, feature_attention_mask,
                     attention_mask, labels=None):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if input_features is None or input_ids.shape[1] == 1:
        return inputs_embeds
    audio_feat_lengths, audio_output_lengths = model.audio_tower._get_feat_extract_output_lengths(
        feature_attention_mask.sum(-1))
    batch_size, _, max_mel_seq_len = input_features.shape
    max_seq_len = (max_mel_seq_len - 2) // 2 + 1
    seq_range = torch.arange(
        max_seq_len, dtype=audio_feat_lengths.dtype,
        device=audio_feat_lengths.device).unsqueeze(0).expand(batch_size, max_seq_len)
    padding_mask = seq_range >= audio_feat_lengths.unsqueeze(1)
    audio_attention_mask_ = padding_mask.view(
        batch_size, 1, 1, max_seq_len).expand(batch_size, 1, max_seq_len, max_seq_len)
    audio_attention_mask = audio_attention_mask_.to(
        dtype=model.audio_tower.conv1.weight.dtype,
        device=model.audio_tower.conv1.weight.device)
    audio_attention_mask[audio_attention_mask_] = float("-inf")
    audio_outputs = model.audio_tower(input_features, attention_mask=audio_attention_mask)
    audio_features = model.multi_modal_projector(audio_outputs.last_hidden_state)
    inputs_embeds, _, _, _, _ = model._merge_input_ids_with_audio_features(
        audio_features, audio_output_lengths, inputs_embeds, input_ids,
        attention_mask, labels)
    return inputs_embeds


def get_audio_file_list(directory):
    def sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)
    return sorted(
        (os.path.join(directory, name) for name in os.listdir(directory)
         if name.lower().endswith(".wav")), key=sort_key)


def get_json_pairs(path, key1, key2):
    with open(path, encoding="utf-8") as stream:
        return [(row.get(key1, ""), row.get(key2, ""))
                for line in stream if line.strip() for row in [json.loads(line)]]


def extract_single_quoted_text(text):
    if "'" not in text:
        return text
    match = re.search(r"'(.*?)'", text)
    end = text.rfind("'")
    return text[match.start() + 1:end] if match and end > match.start() else text


def _edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def calc_asr_metrics(results):
    punctuation = re.compile(r"[.,!?\"'()\-]")
    values = []
    for reference, hypothesis in results:
        if not hypothesis or hypothesis == "NA":
            continue
        reference = punctuation.sub("", reference).lower()
        hypothesis = punctuation.sub("", hypothesis).lower()
        ref_words, hyp_words = reference.split(), hypothesis.split()
        wer = _edit_distance(ref_words, hyp_words) / max(1, len(ref_words))
        cer = _edit_distance(reference, hypothesis) / max(1, len(reference))
        values.append((cer, wer))
    if not values:
        raise ValueError("no valid ASR hypotheses")
    return {
        "n": len(values),
        "cer": sum(item[0] for item in values) / len(values),
        "wer": sum(item[1] for item in values) / len(values),
    }


def pool_time(tensor, block_size):
    if not block_size or block_size <= 1:
        return tensor
    pad = (-tensor.shape[1]) % block_size
    if pad:
        tensor = F.pad(tensor, (0, pad))
    return tensor.reshape(tensor.shape[0], -1, block_size).mean(dim=-1)


def compute_saliency_sets_2d(model, asr_model, inputs, target_ids,
                             time_block_size=None):
    """Return absolute refusal and ASR gradient maps for one audio."""
    feature = inputs["input_features"].clone().detach().requires_grad_(True).to(model.device)
    embeds = get_input_embeds(
        model, inputs["input_ids"], feature, inputs["feature_attention_mask"],
        inputs["attention_mask"])
    logits = model(inputs_embeds=embeds, use_cache=False).logits
    shift = embeds.size(1) - target_ids.size(1)
    target_logits = logits[..., shift - 1:-1, :].contiguous()
    refusal_loss = F.cross_entropy(
        target_logits.view(-1, target_logits.size(-1)), target_ids.view(-1))
    refusal_gradient = torch.autograd.grad(
        refusal_loss, feature)[0].abs().squeeze(0)
    del embeds, logits, target_logits, refusal_loss, feature
    torch.cuda.empty_cache()

    from whisper.audio import N_FRAMES, pad_or_trim
    feature = pad_or_trim(
        inputs["input_features"].squeeze(0), N_FRAMES).unsqueeze(0).to(asr_model.device)
    with torch.no_grad():
        decoded = whisper.decode(asr_model, feature, whisper.DecodingOptions())
        predicted = torch.tensor(decoded[0].tokens, device=feature.device).unsqueeze(0)
    feature.requires_grad_(True)
    logits = asr_model(feature, tokens=predicted)
    shift = logits.size(1) - predicted.size(1)
    target_logits = logits[..., shift:, :].contiguous()
    asr_loss = F.cross_entropy(
        target_logits.view(-1, target_logits.size(-1)), predicted.view(-1))
    asr_gradient = torch.autograd.grad(asr_loss, feature)[0].abs().squeeze(0)
    length = min(refusal_gradient.shape[1], asr_gradient.shape[1])
    refusal = pool_time(refusal_gradient[:, :length], time_block_size).cpu().numpy()
    asr = pool_time(asr_gradient[:, :length], time_block_size).cpu().numpy()
    del logits, target_logits, asr_loss, feature, refusal_gradient, asr_gradient
    torch.cuda.empty_cache()
    return refusal, asr
