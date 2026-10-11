"""Loads the trained WF-Diff model. Never substitutes random weights."""
import gc, logging, os, time
import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger("wfdiff")
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(HERE, "models", "best_model_wfdiff.pth")
HF_MODEL_REPO = "Shylesh1640/wfdiff-model"
HF_MODEL_FILE = "best_model_wfdiff.pth"
IMG_SIZE = 256
MODEL_KWARGS: dict = {
    "in_channel": 6,
    "out_channel": 3,
    "inner_channel": 48,
    "norm_groups": 24,
    "with_time_emb": True,
    "schedule_opt": {
        "schedule": "linear",
        "n_timestep": 2000,
        "linear_start": 1e-6,
        "linear_end": 1e-2,
    },
    "sample_proc": "ddim",
}

class ModelLoadError(RuntimeError): pass

def _import_arch():
    try:
        from model_architecture.wfdiff import WfDiffx2, DWT, IWT
    except ImportError as e:
        raise ModelLoadError(
            "WF-Diff architecture not found."
        ) from e
    return WfDiffx2, DWT, IWT

def _ensure_checkpoint():
    if not os.path.isfile(CHECKPOINT):
        log.info("Checkpoint not found locally, downloading from HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download
            os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
            hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILE, local_dir=os.path.dirname(CHECKPOINT))
            log.info("Checkpoint downloaded successfully")
        except Exception as e:
            raise ModelLoadError(f"Cannot download checkpoint: {e}") from e
    if not os.path.isfile(CHECKPOINT):
        raise ModelLoadError(f"Checkpoint missing: models/{os.path.basename(CHECKPOINT)}")

def _load_fresh_model(device="cpu"):
    """Load a fresh model instance each time (no caching) to enable memory-staged inference."""
    _ensure_checkpoint()
    WfDiffx2, _, _ = _import_arch()
    model = WfDiffx2(**MODEL_KWARGS)
    try:
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ModelLoadError(f"Cannot read checkpoint: {e}") from e
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    del ckpt
    gc.collect()
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except TypeError:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise ModelLoadError(f"Checkpoint incompatible with WfDiffx2: {str(e)[:500]}") from e
    del state
    gc.collect()
    model = model.to(device).eval()
    gc.collect()
    log.info("Loaded WF-Diff on %s (%.2fM params)", device, sum(p.numel() for p in model.parameters()) / 1e6)
    return model

def _free_module(parent, attr_name):
    """Delete a sub-module from the parent to free its parameters."""
    if hasattr(parent, attr_name):
        delattr(parent, attr_name)
    gc.collect()

def _mem_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        return -1

@torch.inference_mode()
def _staged_infer(model, x, device):
    """Run WF-Diff inference in stages, freeing each sub-module after use to reduce peak memory."""
    _, DWT_cls, IWT_cls = _import_arch()
    dwt, idwt = DWT_cls(), IWT_cls()

    log.info("Stage 1: init_predictor, mem=%.0f MB", _mem_mb())
    x_, _, _ = model.init_predictor(x)
    _free_module(model, 'init_predictor')

    n = x_.shape[0]
    input_dwt = dwt(x_)
    input_LL, input_high0 = input_dwt[:n], input_dwt[n:]
    del input_dwt

    x_HH, x_LL = model.cfc(input_LL, input_high0)
    _free_module(model, 'cfc')
    gc.collect()

    log.info("Stage 2: denoiser1 (LL), mem=%.0f MB", _mem_mb())
    noisell, _ = model.denoiser1(input_LL, None, x_LL)
    del x_LL
    _free_module(model, 'denoiser1')

    log.info("Stage 3: denoiser2 (high), mem=%.0f MB", _mem_mb())
    noisehigh, _ = model.denoiser2(input_high0, None, x_HH)
    del x_HH
    _free_module(model, 'denoiser2')

    log.info("Stage 4: reconstruct, mem=%.0f MB", _mem_mb())
    out1_dwt = dwt(x_)
    del x_
    out1_LL, out1_high0 = out1_dwt[:n], out1_dwt[n:]
    del out1_dwt
    result = idwt(torch.cat((out1_LL + noisell, out1_high0 + noisehigh), dim=0))
    del out1_LL, out1_high0, noisell, noisehigh, input_LL, input_high0
    gc.collect()
    log.info("Done, mem=%.0f MB", _mem_mb())
    return result

def enhance(image: Image.Image, device="cpu") -> tuple[Image.Image, dict]:
    """Runs inference with staged memory management."""
    t0 = time.time()
    rgb = np.array(image.convert("RGB"))
    rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    I = torch.from_numpy(rgb_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    model = _load_fresh_model(device)
    try:
        out = _staged_infer(model, I, device)
    finally:
        del model
        gc.collect()

    if torch.is_tensor(out):
        out_np = (out.squeeze(0).detach().clamp(0, 1).permute(1, 2, 0).cpu().float().numpy() * 255.0).round().astype(np.uint8)
    else:
        out_np = out
    out_img = Image.fromarray(out_np) if isinstance(out_np, np.ndarray) else out_np
    ms = int((time.time() - t0) * 1000)
    log.info("inference %d ms", ms)
    return out_img, {"latency_ms": ms, "output_width": out_img.width, "output_height": out_img.height}
