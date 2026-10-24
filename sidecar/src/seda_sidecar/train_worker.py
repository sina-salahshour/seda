"""OmniVoice LoRA fine-tuning worker (run as a subprocess by ``training``).

    python -m seda_sidecar.train_worker <run_dir>

Reads ``<run_dir>/job.json`` (written by training.start_run):
    {"profile_id", "engine", "samples": [{"id", "wav", "text", "duration"}],
     "params": {...}, "init_adapter": str|null}

Stages
  1. data: every sample is normalised (text_fa) and encoded with the Higgs audio
     tokenizer exactly like ``omnivoice.scripts.extract_audio_tokens`` (peak-normalised
     to 0.9, int16 [8, T] tokens) and packed into the OmniVoice WebDataset layout:
     data/tokens/train/{audios/shard-000000.tar, txts/shard-000000.jsonl, data.lst}
     + data/data_config.json.
  2. train: model, tokenizer and dataloader come from OmniVoice's own
     ``omnivoice.training.builder`` (build_model_and_tokenizer, then PEFT LoRA exactly as
     upstream ``_apply_lora`` / docs/lora_finetuning.md: r16/alpha32 on q,k,v,o,gate,up,down
     + modules_to_save audio_embeddings/audio_heads; OmniVoiceSampleProcessor, StreamLengthGroupDataset +
     PaddingDataCollator for sdpa). The loop mirrors ``OmniTrainer.train``
     (AdamW over trainable params, cosine schedule with warmup, grad-clip 1.0,
     grad accumulation, bf16 autocast) but without accelerate's full-state
     checkpoints (2.3 GB each) and with gradient checkpointing to fit 8 GB.
  3. save: PEFT adapter (adapter_config.json + adapter_model.safetensors with the
     LoRA weights and the fully-trained audio_embeddings/audio_heads) to run_dir/adapter.
  4. pocket (profile.engine == "pocket_v2"): engines.pocket_engine.prepare_pocket_reference.

Progress is reported on stdout as lines ``@@PROGRESS {json}`` which the parent
(training.py) turns into run.json updates; everything else is plain log text.
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROGRESS_PREFIX = "@@PROGRESS "
DATA_FRACTION = 0.08  # share of the progress bar used by data preparation


def emit(**kw) -> None:
    print(PROGRESS_PREFIX + json.dumps(kw, ensure_ascii=False), flush=True)


_T0 = time.time()


def logp(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- data
def prepare_data(job: dict, run_dir: Path, device: str) -> Path:
    import numpy as np
    import soundfile as sf
    import torch
    import webdataset as wds
    from omnivoice.models.omnivoice import _resolve_model_path
    from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel

    from . import config

    try:
        from .text_fa import normalize_fa_text
    except ImportError:  # pragma: no cover
        def normalize_fa_text(t: str) -> str:
            return " ".join(t.split())

    data_dir = run_dir / "data"
    out = data_dir / "tokens" / "train"
    (out / "audios").mkdir(parents=True, exist_ok=True)
    (out / "txts").mkdir(parents=True, exist_ok=True)

    model_path = _resolve_model_path(config.OMNIVOICE_MODEL)
    tok_path = os.path.join(model_path, "audio_tokenizer")
    if not os.path.isdir(tok_path):
        tok_path = _resolve_model_path("eustlb/higgs-audio-v2-tokenizer")
    logp(f"loading audio tokenizer {tok_path} on {device}")
    fe = AutoFeatureExtractor.from_pretrained(tok_path)
    tok = HiggsAudioV2TokenizerModel.from_pretrained(tok_path, device_map=device)
    sr_tok = 24_000

    samples = list(job["samples"])
    random.Random(0).shuffle(samples)
    tar_path = out / "audios" / "shard-000000.tar"
    jsonl_path = out / "txts" / "shard-000000.jsonl"
    n_ok, total_dur = 0, 0.0
    with wds.TarWriter(str(tar_path)) as tw, open(jsonl_path, "w", encoding="utf-8") as jf:
        for i, s in enumerate(samples):
            text = normalize_fa_text(s.get("text") or "").strip()
            if not text:
                logp(f"skip {s['id']}: empty transcript")
                continue
            try:
                wav, sr = sf.read(s["wav"], dtype="float32", always_2d=True)
                wav = wav.mean(axis=1)
                if sr != sr_tok:
                    import librosa

                    wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_tok)
                wav = wav / (np.abs(wav).max() + 1e-7) * 0.9
                with torch.inference_mode():
                    inputs = fe(raw_audio=wav, sampling_rate=sr_tok, return_tensors="pt").to(tok.device)
                    codes = tok.encode(inputs["input_values"]).audio_codes.squeeze(0)
                assert codes.dim() == 2 and codes.size(0) == 8, tuple(codes.shape)
                arr = codes.to(torch.int16).cpu().numpy()
            except Exception as e:  # noqa: BLE001
                logp(f"skip {s['id']}: {e}")
                continue
            dur = len(wav) / sr_tok
            buf = io.BytesIO()
            np.save(buf, arr)
            tw.write({"__key__": s["id"], "npy": buf.getvalue()})
            meta = {
                "id": s["id"],
                "text": text,
                "language_id": "fa",
                "audio_duration": round(dur, 3),
                "num_tokens": int(arr.shape[1]),
            }
            jf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            n_ok += 1
            total_dur += dur
            emit(stage="data", progress=DATA_FRACTION * (i + 1) / len(samples),
                 message=f"preparing data {i + 1}/{len(samples)}")
    del tok, fe
    from .engines.omnivoice_engine import _clear_higgs_caches

    _clear_higgs_caches()  # lru_cache in transformers would keep the tokenizer on the GPU
    import gc

    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if n_ok == 0:
        raise RuntimeError("no usable samples (all empty or unreadable)")
    (out / "data.lst").write_text(
        f"{tar_path.resolve()} {jsonl_path.resolve()} {n_ok} {total_dur:.3f}\n", encoding="utf-8"
    )
    data_cfg = data_dir / "data_config.json"
    data_cfg.write_text(json.dumps({"train": [{"manifest_path": [str((out / "data.lst").resolve())]}]}))
    logp(f"prepared {n_ok} samples, {total_dur / 60:.1f} min of audio")
    return data_cfg


# ---------------------------------------------------------------- train
def train(job: dict, run_dir: Path, data_cfg: Path, device: str) -> None:
    import torch
    from omnivoice.training import builder
    from omnivoice.training.config import TrainingConfig
    from transformers import get_cosine_schedule_with_warmup

    from . import config

    p = job["params"]
    steps = int(p["steps"])
    accum = int(p.get("grad_accum", 1))
    init_adapter = job.get("init_adapter")
    tc = TrainingConfig(
        output_dir=str(run_dir / "work"),
        data_config=str(data_cfg),
        init_from_checkpoint=config.OMNIVOICE_MODEL,
        learning_rate=float(p.get("learning_rate", 1e-4)),
        weight_decay=0.01,
        max_grad_norm=1.0,
        steps=steps,
        seed=int(p.get("seed", 42)),
        warmup_ratio=float(p.get("warmup_ratio", 0.05)),
        # same conditioning mix as examples/config/train_config_finetune_lora.json
        drop_cond_ratio=0.1,
        prompt_ratio_range=(0.0, 0.3),
        mask_ratio_range=(0.0, 1.0),
        language_ratio=0.8,
        use_pinyin_ratio=0.0,
        instruct_ratio=0.0,
        only_instruct_ratio=0.0,
        attn_implementation="sdpa",
        batch_tokens=int(p.get("batch_tokens", 3072)),
        max_sample_tokens=int(p.get("max_sample_tokens", 1500)),
        min_sample_tokens=int(p.get("min_sample_tokens", 30)),
        max_batch_size=int(p.get("max_batch_size", 16)),
        gradient_accumulation_steps=accum,
        num_workers=1,
        mixed_precision="bf16",
    )
    random.seed(tc.seed)
    torch.manual_seed(tc.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    emit(stage="model", progress=DATA_FRACTION, message="loading model")
    logp("loading OmniVoice for training")
    model, tokenizer = builder.build_model_and_tokenizer(tc)
    if init_adapter:
        from peft import PeftModel

        logp(f"initialising from adapter {init_adapter}")
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        # Same as upstream omnivoice.training.builder._apply_lora (newer than the
        # PyPI 0.2.1 release, which has no use_lora option).
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=int(p.get("lora_r", 16)),
            lora_alpha=int(p.get("lora_alpha", 32)),
            lora_dropout=float(p.get("lora_dropout", 0.05)),
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["audio_embeddings", "audio_heads"],
        ))
    model.print_trainable_parameters()
    base = model.get_base_model()
    if p.get("gradient_checkpointing", True):
        base.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.llm.config.use_cache = False
    model.to(device)
    model.train()

    logp("model ready")
    train_loader, _ = builder.build_dataloaders(tc, tokenizer)

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc.learning_rate, weight_decay=tc.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(steps * tc.warmup_ratio), steps)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.autocast("cpu", dtype=torch.bfloat16)

    epoch = 0
    if hasattr(train_loader.dataset, "set_epoch"):
        train_loader.dataset.set_epoch(epoch)
    it = iter(train_loader)

    def next_batch():
        nonlocal it, epoch
        try:
            return next(it)
        except StopIteration:
            epoch += 1
            train_loader.dataset.set_epoch(epoch)
            it = iter(train_loader)
            return next(it)

    step, ema = 0, None
    t0 = time.time()
    last_emit = 0.0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    emit(stage="train", step=0, total_steps=steps, progress=DATA_FRACTION + 0.02, message="training")
    while step < steps:
        loss_acc = 0.0
        for _ in range(accum):
            batch = next_batch()
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with amp:
                out = model(**batch)
            loss = out.loss / accum
            loss.backward()
            loss_acc += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, tc.max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        ema = loss_acc if ema is None else 0.9 * ema + 0.1 * loss_acc
        now = time.time()
        if now - last_emit > 1.0 or step == steps or step <= 3:
            last_emit = now
            sps = (now - t0) / step
            frac = DATA_FRACTION + 0.02 + (1 - DATA_FRACTION - 0.04) * step / steps
            peak = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else 0
            eta = sps * (steps - step)
            emit(stage="train", step=step, total_steps=steps, loss=round(ema, 4), progress=round(frac, 4),
                 sec_per_step=round(sps, 3), peak_vram_mb=int(peak), epoch=epoch,
                 message=f"training step {step}/{steps}, ~{int(eta // 60)}m{int(eta % 60):02d}s left")
            logp(f"step {step}/{steps} loss {loss_acc:.4f} (ema {ema:.4f}) lr {sched.get_last_lr()[0]:.2e} "
                 f"{sps:.2f}s/step epoch {epoch} peak {peak:.0f}MB")

    emit(stage="save", step=steps, total_steps=steps, progress=0.97, message="saving adapter")
    adapter_dir = run_dir / "adapter"
    tmp = run_dir / "adapter.tmp"
    model.save_pretrained(str(tmp))
    if adapter_dir.exists():
        import shutil

        shutil.rmtree(adapter_dir)
    tmp.rename(adapter_dir)
    logp(f"saved adapter to {adapter_dir} ({(time.time() - t0):.0f}s training)")
    del model, opt, base
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def pocket_reference(job: dict, run_dir: Path) -> None:
    try:
        from . import store
        from .engines.pocket_engine import prepare_pocket_reference
    except ImportError as e:
        logp(f"pocket reference skipped: {e}")
        return
    emit(stage="pocket", progress=0.98, message="choosing pocket reference")
    info = prepare_pocket_reference(store.voice_profile(job["profile_id"]), run_dir / "pocket")
    logp(f"pocket reference: {info}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m seda_sidecar.train_worker <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1]).resolve()
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logp(f"seda train_worker: run {run_dir.name}, {len(job['samples'])} samples, device {device}, "
         f"params {json.dumps(job['params'])}")
    emit(stage="data", progress=0.0, message="preparing data")
    data_cfg = prepare_data(job, run_dir, device)
    train(job, run_dir, data_cfg, device)
    if job.get("engine") == "pocket_v2":
        pocket_reference(job, run_dir)
    emit(stage="done", progress=1.0, message="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
