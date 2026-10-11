import base64, gc, io, json, logging, os, time, threading, uuid
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
jobs = {}

def _mem_mb():
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

def _run_job(job_id, image):
    try:
        jobs[job_id]["status"] = "processing"
        out, info = model_loader.enhance(image, "cpu")
        buf = io.BytesIO()
        out.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        gc.collect()
        jobs[job_id] = {
            "status": "done",
            "image": f"data:image/png;base64,{img_b64}",
            "info": info,
        }
    except Exception as e:
        log.error("Job %s failed: %s", job_id, e)
        jobs[job_id] = {"status": "error", "error": str(e)}

@app.route('/enhance', methods=['POST'])
def enhance():
    log.info("Request received, mem=%.0f MB", _mem_mb())
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued"}
    threading.Thread(target=_run_job, args=(job_id, image), daemon=True).start()
    return jsonify({"job_id": job_id, "status": "queued"})

@app.route('/result/<job_id>')
def get_result(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] == "done":
        result = dict(job)
        del jobs[job_id]
        return jsonify(result)
    return jsonify({"status": job["status"]})

if __name__ == '__main__':
    log.info("Starting app, mem=%.0f MB", _mem_mb())
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, threaded=True)
