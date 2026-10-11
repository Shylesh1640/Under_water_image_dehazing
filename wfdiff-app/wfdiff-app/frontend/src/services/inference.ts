const BACKEND_URL =
  (import.meta.env.VITE_HF_SPACE_ID as string | undefined) ||
  window.location.origin;

export const MAX_BYTES = 10 * 1024 * 1024;
export const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];

export interface EnhanceResult {
  url: string;
  latencyMs: number | null;
  width: number | null;
  height: number | null;
}
export class InferenceError extends Error {
  constructor(public kind: string, message: string) {
    super(message);
  }
}

export async function checkService(): Promise<boolean> {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export function classifyError(e: unknown): InferenceError {
  if (e instanceof InferenceError) return e;
  const raw = (e as any)?.message ?? (e as any)?.title ?? String(e);
  const m = String(raw).toLowerCase();
  if (m.includes("out of memory"))
    return new InferenceError("memory", "The server ran out of memory for this request.");
  if (m.includes("timeout") || m.includes("timed out"))
    return new InferenceError("timeout", "The request timed out. The server may be starting up; retry.");
  if (m.includes("unavailable") || m.includes("fetch") || m.includes("connect"))
    return new InferenceError("unavailable", "The inference service is unavailable or starting up. Retry shortly.");
  return new InferenceError("inference", `Inference failed: ${String(raw).slice(0, 200)}`);
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

export async function enhance(
  file: File,
  onStatus?: (s: string) => void,
): Promise<EnhanceResult> {
  try {
    onStatus?.("Uploading image");
    const form = new FormData();
    form.append("image", file);
    const submitRes = await fetch(`${BACKEND_URL}/enhance`, {
      method: "POST",
      body: form,
    });
    if (!submitRes.ok) {
      const err = await submitRes.json().catch(() => ({ error: `Server error: ${submitRes.status}` }));
      throw new Error(err.error || `Server error: ${submitRes.status}`);
    }
    const { job_id } = await submitRes.json();
    if (!job_id) throw new InferenceError("output", "Server did not return a job ID.");

    onStatus?.("Processing (this may take several minutes on CPU)");
    for (let i = 0; i < 200; i++) {
      await sleep(3000);
      try {
        const pollRes = await fetch(`${BACKEND_URL}/result/${job_id}`);
        const data = await pollRes.json();
        if (data.status === "done") {
          if (!data.image) throw new InferenceError("output", "No image in result.");
          return {
            url: data.image,
            latencyMs: data.info?.latency_ms ?? null,
            width: data.info?.output_width ?? null,
            height: data.info?.output_height ?? null,
          };
        }
        if (data.status === "error") {
          throw new Error(data.error || "Inference failed on server.");
        }
        const mins = Math.floor((i * 3) / 60);
        const secs = (i * 3) % 60;
        onStatus?.(`Processing${mins > 0 ? ` (${mins}m ${secs}s)` : ` (${secs}s)`}`);
      } catch (e) {
        if (e instanceof InferenceError) throw e;
        if ((e as Error).message?.includes("Inference failed")) throw classifyError(e);
      }
    }
    throw new InferenceError("timeout", "Processing took too long. Please try again.");
  } catch (e) {
    throw classifyError(e);
  }
}
