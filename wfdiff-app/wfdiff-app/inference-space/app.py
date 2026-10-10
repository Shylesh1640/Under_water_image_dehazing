import json, logging, os
import gradio as gr
import torch
from PIL import Image
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

try:
    import spaces
    gpu = spaces.GPU(duration=120)
except Exception:
    def gpu(fn): return fn

import model_loader

logging.basicConfig(level=logging.INFO)
MAX_SIDE = 4096
HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")

@gpu
def enhance(image: Image.Image):
    if image is None:
        raise gr.Error("No image provided.")
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        out, info = model_loader.enhance(image, device)
    except model_loader.ModelLoadError as e:
        raise gr.Error(f"Model error: {e}")
    return out, json.dumps(info)

with gr.Blocks(title="WF-Diff") as demo:
    gr.Markdown("# WF-Diff underwater image restoration\nOutput is 256x256.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Underwater image", format="png")
        out = gr.Image(type="pil", label="Enhanced", format="png")
    info = gr.Textbox(label="Run info (JSON)")
    gr.Button("Enhance Image", variant="primary").click(enhance, inp, [out, info], api_name="enhance")

app = FastAPI()

# Mount Gradio at /gradio (API available at /gradio/gradio_api/*)
app = gr.mount_gradio_app(app, demo, path="/gradio")

# Serve React frontend at root
if os.path.isdir(STATIC_DIR):
    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"\n  React frontend: http://localhost:{port}/")
    print(f"  Gradio UI:      http://localhost:{port}/gradio")
    print(f"  Gradio API:     http://localhost:{port}/gradio/gradio_api/\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
