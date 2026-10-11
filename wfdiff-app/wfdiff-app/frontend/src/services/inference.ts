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
  if (m.includes("quota") || m.includes("zerogpu"))
    return new InferenceError("quota", "The free GPU quota is used up. Wait a few minutes and retry.");
  if (m.includes("checkpoint"))
    return new InferenceError("model", "The model checkpoint on the server is invalid or missing.");
  if (m.includes("out of memory"))
    return new InferenceError("memory", "The server ran out of memory for this request.");
  if (m.includes("timeout") || m.includes("timed out"))
    return new InferenceError("timeout", "The request timed out. The server may be starting up; retry.");
  if (m.includes("queue") || m.includes("unavailable") || m.includes("fetch") || m.includes("connect"))
    return new InferenceError("unavailable", "The inference service is unavailable or starting up. Retry shortly.");
  return new InferenceError("inference", `Inference failed: ${String(raw).slice(0, 200)}`);
}

export async function enhance(
  file: File,
  onStatus?: (s: string) => void,
): Promise<EnhanceResult> {
  try {
    onStatus?.("Uploading image");
    const form = new FormData();
    form.append("image", file);
    const res = await fetch(`${BACKEND_URL}/enhance`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `Server error: ${res.status}` }));
      throw new Error(err.error || `Server error: ${res.status}`);
    }
    onStatus?.("Processing");
    const data = await res.json();
    if (!data.image) throw new InferenceError("output", "The service returned no valid image.");
    return {
      url: data.image,
      latencyMs: data.info?.latency_ms ?? null,
      width: data.info?.output_width ?? null,
      height: data.info?.output_height ?? null,
    };
  } catch (e) {
    throw classifyError(e);
  }
}
