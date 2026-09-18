"""Kimi-Audio helpers used by the 2D saliency and mixed-loss training paths."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import whisper  # noqa: E402
from utils import pool_time  # noqa: E402
from kimi_adapter import build_prompt_tokens, extract_mel, kimi_jb_logits  # noqa: E402

REFUSAL = "I'm sorry, but I cannot fulfill this request."


def prepare_inputs_and_targets_kimi(audio_path, feature_extractor, tokenizer,
                                    extra_tokens, device, target_text=REFUSAL,
                                    question=None):
    mel, token_len = extract_mel(feature_extractor, audio_path, device=device)
    audio_ids, text_ids, continuous_mask, target_ids = build_prompt_tokens(
        extra_tokens, tokenizer, token_len, device,
        question=question, target_text=target_text)
    return mel, audio_ids, text_ids, continuous_mask, token_len, target_ids


def _asr_gradient(asr_model, mel):
    from whisper.audio import N_FRAMES, pad_or_trim
    feature = pad_or_trim(mel.squeeze(0), N_FRAMES).unsqueeze(0).to(asr_model.device)
    with torch.no_grad():
        decoded = whisper.decode(asr_model, feature, whisper.DecodingOptions())
        predicted = torch.tensor(decoded[0].tokens, device=feature.device).unsqueeze(0)
    feature.requires_grad_(True)
    logits = asr_model(feature, tokens=predicted)
    shift = logits.size(1) - predicted.size(1)
    target_logits = logits[..., shift:, :].contiguous()
    loss = F.cross_entropy(
        target_logits.view(-1, target_logits.size(-1)), predicted.view(-1))
    gradient = torch.autograd.grad(loss, feature)[0].abs().squeeze(0).cpu().numpy()
    del logits, target_logits, loss, feature
    torch.cuda.empty_cache()
    return gradient


def _refusal_gradient(model, whisper_encoder, mel, audio_ids, text_ids,
                      continuous_mask, token_len, target_ids):
    logits = kimi_jb_logits(
        model, whisper_encoder, mel, audio_ids, text_ids,
        continuous_mask, token_len)
    shift = logits.size(1) - target_ids.size(0)
    target_logits = logits[..., shift - 1:-1, :].contiguous()
    loss = F.cross_entropy(
        target_logits.view(-1, target_logits.size(-1)), target_ids.view(-1))
    gradient = torch.autograd.grad(loss, mel)[0].abs().squeeze(0).cpu().numpy()
    del logits, target_logits, loss
    torch.cuda.empty_cache()
    return gradient


def compute_saliency_maps_2d(model, whisper_encoder, asr_model, mel, audio_ids,
                             text_ids, continuous_mask, token_len, target_ids,
                             time_block_size=None):
    mel = mel.clone().detach().requires_grad_(True)
    refusal = _refusal_gradient(
        model, whisper_encoder, mel, audio_ids, text_ids,
        continuous_mask, token_len, target_ids)
    asr = _asr_gradient(asr_model, mel)
    return (pool_time(torch.from_numpy(refusal), time_block_size).numpy(),
            pool_time(torch.from_numpy(asr), time_block_size).numpy())
