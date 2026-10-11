import { useCallback, useEffect, useRef, useState } from "react";
import { Upload, Waves, Download, Loader2, RefreshCw, Columns2, SlidersHorizontal, AlertTriangle } from "lucide-react";
import { ACCEPTED, MAX_BYTES, checkService, enhance, InferenceError, type EnhanceResult } from "./services/inference";

const fmt = (b: number) => (b > 1048576 ? `${(b / 1048576).toFixed(2)} MB` : `${(b / 1024).toFixed(0)} KB`);

function Compare({ before, after, side }: { before: string; after: string; side: boolean }) {
  const [pos, setPos] = useState(50);
  if (side)
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        <figure><img src={before} alt="Original" className="w-full rounded-lg" /><figcaption className="mt-1 text-sm text-slate-400">Original</figcaption></figure>
        <figure><img src={after} alt="Enhanced" className="w-full rounded-lg" /><figcaption className="mt-1 text-sm text-slate-400">Enhanced</figcaption></figure>
      </div>
    );
  return (
    <div className="relative mx-auto max-w-2xl select-none overflow-hidden rounded-lg">
      <img src={after} alt="Enhanced" className="block w-full" />
      <img src={before} alt="Original" className="absolute inset-0 h-full w-full object-fill" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
      <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-cyan-300" style={{ left: `${pos}%` }} />
      <input type="range" min={0} max={100} value={pos} onChange={(e) => setPos(+e.target.value)} aria-label="Before and after comparison position" className="absolute inset-0 h-full w-full cursor-ew-resize opacity-0" />
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dims, setDims] = useState<[number, number] | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<(EnhanceResult & { bytes: number; blobUrl: string }) | null>(null);
  const [side, setSide] = useState(false);
  const [service, setService] = useState<"checking" | "up" | "down">("checking");
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => { checkService().then((ok) => setService(ok ? "up" : "down")); }, []);

  const pick = useCallback((f: File | undefined) => {
    if (!f) return;
    setError(null); setResult(null);
    if (!ACCEPTED.includes(f.type)) return setError("Unsupported format. Use JPEG, PNG or WEBP.");
    if (f.size > MAX_BYTES) return setError(`File is ${fmt(f.size)}. The limit is ${fmt(MAX_BYTES)}.`);
    const url = URL.createObjectURL(f);
    const im = new Image(); im.onload = () => setDims([im.naturalWidth, im.naturalHeight]); im.src = url;
    setFile(f); setPreview(url);
  }, []);

  const run = async () => {
    if (!file) return setError("Select an image first.");
    if (busy) return;
    setBusy(true); setError(null); setResult(null); setStatus("Sending image");
    try {
      const r = await enhance(file, setStatus);
      const blob = await (await fetch(r.url)).blob();
      const png = blob.type === "image/png" ? blob : await toPng(blob);
      setResult({ ...r, bytes: png.size, blobUrl: URL.createObjectURL(png) });
    } catch (e) { setError((e as InferenceError).message); }
    finally { setBusy(false); setStatus(""); }
  };

  return (
    <main className="mx-auto max-w-5xl px-4 py-10">
      <header className="mb-8">
        <div className="flex items-center gap-3"><Waves className="text-cyan-300" aria-hidden /><h1 className="text-2xl font-extrabold sm:text-3xl">WF-Diff | Underwater Image Restoration</h1></div>
        <p className="mt-3 max-w-2xl text-slate-300">Underwater photos lose contrast and colour to scattering and absorption. WF-Diff restores them with a wavelet-domain diffusion model. Output is 256 × 256.</p>
        <p className="mt-2 text-sm" role="status">
          <span className={`mr-2 inline-block h-2 w-2 rounded-full ${service === "up" ? "bg-emerald-400" : service === "down" ? "bg-rose-400" : "bg-slate-500"}`} />
          {service === "up" ? "Inference service reachable" : service === "down" ? "Inference service unavailable" : "Checking service"}
        </p>
      </header>

      <section className="panel p-5" aria-label="Upload">
        <div onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); pick(e.dataTransfer.files[0]); }}
          className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-cyan-300/30 p-8 text-center">
          <Upload className="text-cyan-300" aria-hidden />
          <p>Drop an image here, or</p>
          <button className="rounded-lg bg-reef px-4 py-2 font-semibold hover:bg-cyan-900" onClick={() => input.current?.click()}>{file ? "Replace image" : "Choose image"}</button>
          <input ref={input} type="file" accept=".jpg,.jpeg,.png,.webp" className="hidden" aria-label="Choose an image file" onChange={(e) => pick(e.target.files?.[0])} />
          <p className="text-xs text-slate-400">JPEG, PNG or WEBP, up to {fmt(MAX_BYTES)}</p>
        </div>
        {preview && file && (
          <div className="mt-4 flex flex-col gap-4 sm:flex-row sm:items-center">
            <img src={preview} alt="Original upload preview" className="max-h-48 rounded-lg" />
            <p className="text-sm text-slate-300">{file.name}<br />{dims ? `${dims[0]} × ${dims[1]} px` : ""} · {fmt(file.size)}</p>
          </div>
        )}
        <button onClick={run} disabled={busy || !file} className="mt-5 inline-flex items-center gap-2 rounded-lg bg-cyan-400 px-5 py-2.5 font-bold text-abyss disabled:cursor-not-allowed disabled:opacity-50">
          {busy ? <Loader2 className="animate-spin" size={18} aria-hidden /> : <Waves size={18} aria-hidden />}Enhance Image
        </button>
        {busy && <p className="mt-3 text-sm text-slate-300" role="status">{status}. Diffusion sampling can take a while on CPU, and the first request may need to load the model.</p>}
        {error && (
          <div role="alert" className="mt-4 flex items-start gap-3 rounded-lg border border-rose-400/40 bg-rose-950/40 p-3 text-sm">
            <AlertTriangle size={18} className="mt-0.5 shrink-0 text-rose-300" aria-hidden /><span className="flex-1">{error}</span>
            {file && <button onClick={run} className="inline-flex items-center gap-1 underline"><RefreshCw size={14} aria-hidden />Retry</button>}
          </div>
        )}
      </section>

      {result && preview && (
        <section className="panel mt-6 p-5" aria-label="Result">
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <h2 className="text-lg font-bold">Result</h2>
            <button onClick={() => setSide(!side)} className="ml-auto inline-flex items-center gap-2 rounded-lg bg-reef px-3 py-1.5 text-sm">
              {side ? <SlidersHorizontal size={14} aria-hidden /> : <Columns2 size={14} aria-hidden />}{side ? "Slider view" : "Side by side"}
            </button>
            <a href={result.blobUrl} download={`wfdiff_${file?.name.replace(/\.[^.]+$/, "")}.png`} className="inline-flex items-center gap-2 rounded-lg bg-cyan-400 px-3 py-1.5 text-sm font-bold text-abyss"><Download size={14} aria-hidden />Download PNG</a>
          </div>
          <Compare before={preview} after={result.blobUrl} side={side} />
          <p className="mt-3 text-sm text-slate-300">
            {result.width && result.height ? `${result.width} × ${result.height} px` : ""}{` · ${fmt(result.bytes)} PNG`}
            {result.latencyMs != null ? ` · inference ${(result.latencyMs / 1000).toFixed(1)} s` : ""}
          </p>
        </section>
      )}
    </main>
  );
}

async function toPng(blob: Blob): Promise<Blob> {
  const bmp = await createImageBitmap(blob);
  const c = document.createElement("canvas"); c.width = bmp.width; c.height = bmp.height;
  c.getContext("2d")!.drawImage(bmp, 0, 0);
  return new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new InferenceError("output", "Invalid output image"))), "image/png"));
}
