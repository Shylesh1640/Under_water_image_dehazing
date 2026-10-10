"""Loads the trained WF-Diff model. Never substitutes random weights."""
import logging, os, time
import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger("wfdiff")
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(HERE, "models", "best_model_wfdiff.pth")
HF_MODEL_REPO = "Shylesh1640/wfdiff-model"
HF_MODEL_FILE = "best_model_wfdiff.pth"
IMG_SIZE = 256            # documented in notebook; inference resizes with OpenCV
DDIM_STEPS = 10           # documented full-pipeline value
# Documented config from notebook: inner_channel=48, 24 GroupNorm groups, 2000 linear timesteps, 6-ch in / 3-ch out.
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
_model = None

def _import_arch():
    try:
        from model_architecture.wfdiff import WfDiffx2, wfdiff_infer  # extracted from notebook
    except ImportError as e:
        raise ModelLoadError(
            "WF-Diff architecture not found. Extract it from the notebook into "
            "model_architecture/wfdiff.py (see model_architecture/EXTRACT_FROM_NOTEBOOK.md)."
        ) from e
    return WfDiffx2, wfdiff_infer

def load_model(device="cpu"):
    """Build WfDiffx2 and load best_model_wfdiff.pth with strict=True."""
    global _model
    if _model is not None:
        return _model
    if "padiff" in os.path.basename(CHECKPOINT).lower():
        raise ModelLoadError("PADiff checkpoint must not be used.")
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
    WfDiffx2, _ = _import_arch()
    model = WfDiffx2(**MODEL_KWARGS)
    try:
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ModelLoadError(f"Cannot read checkpoint: {e}") from e
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt  # raw state_dict or final format
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise ModelLoadError(f"Checkpoint incompatible with WfDiffx2: {str(e)[:500]}") from e
    _model = model.to(device).eval()
    log.info("Loaded WF-Diff on %s (%.2fM params)", device, sum(p.numel() for p in model.parameters()) / 1e6)
    return _model

def enhance(image: Image.Image, device="cpu") -> tuple[Image.Image, dict]:
    """Runs the notebook's inference pipeline on one image (batch size 1)."""
    _, wfdiff_infer = _import_arch()
    model = load_model(device)
    t0 = time.time()
    rgb = np.array(image.convert("RGB"))
    # Resize with OpenCV to 256x256 matching the notebook's dehaze_image pipeline
    rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    # Convert to 1x3xHxW float32 tensor in [0, 1]
    I = torch.from_numpy(rgb_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    try:
        with torch.inference_mode():
            out = wfdiff_infer(model, I)
    except torch.cuda.OutOfMemoryError as e:
        raise ModelLoadError("GPU out of memory") from e
    if device == "cuda":
        log.info("peak GPU mem: %.0f MB", torch.cuda.max_memory_allocated() / 1e6)
    if torch.is_tensor(out):
        out_np = (out.squeeze(0).detach().clamp(0, 1).permute(1, 2, 0).cpu().float().numpy() * 255.0).round().astype(np.uint8)
    else:
        out_np = out
    out_img = Image.fromarray(out_np) if isinstance(out_np, np.ndarray) else out_np
    ms = int((time.time() - t0) * 1000)
    log.info("inference %d ms", ms)
    return out_img, {"latency_ms": ms, "output_width": out_img.width, "output_height": out_img.height}
