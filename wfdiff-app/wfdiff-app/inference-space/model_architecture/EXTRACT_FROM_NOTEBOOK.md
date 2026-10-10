# Required: copy these from `WF-Diff_UIEB_hybrid(1).ipynb` (no rewriting)

Create `model_architecture/wfdiff.py` containing, verbatim from the notebook:
- `WfDiffx2` and every class/function it depends on (Haar DWT/IDWT, FFT/Fourier interaction, CFC, WFI2-net, denoisers, Gaussian diffusion + DDIM)
- `wfdiff_infer` (or the equivalent inference function) and its helpers

Exclude: dataset loading, augmentation, training loops, optimizers, metrics, logs.
The loader expects `WfDiffx2`, `wfdiff_infer`, and a constructor-kwargs dict matching the notebook.
