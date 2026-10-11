import base64, gc, io, json, logging, os, time
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

import model_loader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MAX_SIDE = 4096

def _mem_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        pass
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        return -1

@app.route('/health')
def health():
    return jsonify({"status": "ok", "mem_mb": round(_mem_mb(), 1)})

@app.route('/enhance', methods=['POST'])
def enhance():
    log.info("Request received, mem=%.0f MB", _mem_mb())
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    try:
        out, info = model_loader.enhance(image, "cpu")
    except model_loader.ModelLoadError as e:
        return jsonify({"error": f"Model error: {e}"}), 500
    log.info("Inference done, mem=%.0f MB", _mem_mb())
    buf = io.BytesIO()
    out.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    gc.collect()
    return jsonify({
        "image": f"data:image/png;base64,{img_b64}",
        "info": info,
    })

if __name__ == '__main__':
    log.info("Starting app, mem=%.0f MB", _mem_mb())
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
