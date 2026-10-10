import { Client, handle_file } from "@gradio/client";

const SPACE_ID = import.meta.env.VITE_HF_SPACE_ID as string | undefined || window.location.origin + "/gradio";
const API_NAME = (import.meta.env.VITE_HF_API_NAME as string | undefined) ?? "/enhance";

export const MAX_BYTES = 10 * 1024 * 1024;
export const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];

export interface EnhanceResult { url: string; latencyMs: number | null; width: number | null; height: number | null; }
export class InferenceError extends Error { constructor(public kind: string, message: string) { super(message); } }

let clientPromise: Promise<Client> | null = null;
function getClient(): Promise<Client> {
  clientPromise ??= Client.connect(SPACE_ID);
  return clientPromise;
}

export async function checkService(): Promise<boolean> {
  try { await getClient(); return true; } catch { clientPromise = null; return false; }
}

export function classifyError(e: unknown): InferenceError {
  if (e instanceof InferenceError) return e;
  const raw = (e as any)?.message ?? (e as any)?.title ?? String(e);
  const m = String(raw).toLowerCase();
  if (m.includes("quota") || m.includes("zerogpu")) return new InferenceError("quota", "The free GPU quota is used up. Wait a few minutes and retry.");
  if (m.includes("checkpoint")) return new InferenceError("model", "The model checkpoint on the server is invalid or missing.");
  if (m.includes("out of memory")) return new InferenceError("memory", "The server ran out of GPU memory for this request.");
  if (m.includes("timeout") || m.includes("timed out")) return new InferenceError("timeout", "The request timed out. The GPU may be starting up; retry.");
  if (m.includes("queue") || m.includes("unavailable") || m.includes("fetch") || m.includes("connect"))
    return new InferenceError("unavailable", "The inference service is unavailable or starting up. Retry shortly.");
  return new InferenceError("inference", `Inference failed: ${String(raw).slice(0, 200)}`);
}

export async function enhance(file: File, onStatus?: (s: string) => void): Promise<EnhanceResult> {
  try {
    onStatus?.("Running on GPU");
    const client = await getClient();
    const result = await client.predict(API_NAME, { image: handle_file(file) });
    const data = result.data as any[];
    if (!data || !data[0]?.url) throw new InferenceError("output", `The service returned no valid image. Got: ${JSON.stringify(data?.[0])?.slice(0, 200)}`);
    const info = typeof data[1] === "string" ? safeJson(data[1]) : data[1];
    return { url: data[0].url, latencyMs: info?.latency_ms ?? null, width: info?.output_width ?? null, height: info?.output_height ?? null };
  } catch (e) { throw classifyError(e); }
}
function safeJson(s: string) { try { return JSON.parse(s); } catch { return null; } }
