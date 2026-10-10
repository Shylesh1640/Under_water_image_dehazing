# WF-Diff web app (scaffold)

**Status:** frontend and Gradio wiring are complete. The model code is NOT included because the notebook and checkpoint were not supplied. Nothing here fakes model output: `inference-space/model_loader.py` raises a clear error until you complete the steps below.

## 1. Finish the inference Space
1. Copy the architecture and `wfdiff_infer` from the notebook into `inference-space/model_architecture/wfdiff.py` (see `EXTRACT_FROM_NOTEBOOK.md`).
2. Set `MODEL_KWARGS` in `model_loader.py` and match the `wfdiff_infer(...)` call to the notebook signature.
3. Put `best_model_wdiff.pth` in `inference-space/models/` (check the notebook's `best_model_wfdiff.pth` name vs your file).
4. Add any extra imports to `requirements.txt`.
5. Local test: `cd inference-space && pip install -r requirements.txt && python app.py`

## 2. Deploy Space A (Gradio, ZeroGPU)
```
huggingface-cli login
huggingface-cli repo create wfdiff-inference --type space --space_sdk gradio
cd inference-space && git init && git lfs install && git lfs track "*.pth"
git remote add space https://huggingface.co/spaces/<user>/wfdiff-inference
git add . && git commit -m init && git push space HEAD:main
```
Select ZeroGPU hardware in Space settings (availability depends on your account). Open "Use via API" and confirm the endpoint is `/enhance`.

## 3. Frontend
```
cd frontend && cp .env.example .env   # set VITE_HF_SPACE_ID=<user>/wfdiff-inference
npm install && npm run dev
npm run build                          # outputs dist/
```
## 4. Deploy Space B (Static)
Create a Static Space, push the contents of `frontend/` (with a README YAML header `sdk: static`, `app_build_command: npm run build`, `app_file: dist/index.html`). Set `VITE_HF_SPACE_ID` as a Space variable.

## Troubleshooting
- "architecture not found": step 1.1 not done.
- "incompatible with WfDiffx2": constructor kwargs differ from training config.
- Quota errors: ZeroGPU quotas are limited; retry later.
- Out of memory: report peak memory from the Space logs.
- Frontend cannot connect: check Space is running and public, and the Space ID.

## Limits
Output is 256 × 256. ZeroGPU runtime/quota fit for a 100M-parameter model with 10 DDIM steps is unverified until tested on the Space.
