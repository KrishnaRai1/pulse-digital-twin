// Thin fetch wrapper: same-origin by default (Vite / nginx proxy /api), optional API key header.
// For a split deployment (e.g. UI on Vercel, API on Render) set VITE_API_BASE=https://api.example.com
// at build time; the WebSocket URL is derived from it unless VITE_WS_URL is given.

export const BASE = ((import.meta.env.VITE_API_BASE as string | undefined) ?? "").replace(/\/+$/, "");
const KEY_STORAGE = "pulse.apiKey";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function getApiKey(): string {
  try {
    return sessionStorage.getItem(KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}
export function setApiKey(key: string): void {
  try {
    if (key) sessionStorage.setItem(KEY_STORAGE, key);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch {
    /* storage unavailable: key is simply not remembered */
  }
}

function detailToText(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const e = d as { loc?: unknown[]; msg?: string };
        const where = (e.loc ?? []).filter((x) => x !== "body").join(".");
        return where ? `${where}: ${e.msg ?? ""}` : (e.msg ?? "");
      })
      .join("; ");
  }
  return fallback;
}

async function request<T>(method: string, path: string, body?: unknown, form?: FormData): Promise<T> {
  const headers: Record<string, string> = {};
  const key = getApiKey();
  if (key && method !== "GET") headers["X-API-Key"] = key;
  let payload: BodyInit | undefined;
  if (form) payload = form;
  else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  let res: Response;
  try {
    res = await fetch(`${BASE}/api/v1${path}`, { method, headers, body: payload });
  } catch {
    throw new ApiError(0, BASE ? `Cannot reach the PULSE API at ${BASE}. It may be waking up (free hosting sleeps when idle): retrying…` : "Cannot reach the PULSE API. Is the backend running?");
  }
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    /* non-JSON error body */
  }
  if (!res.ok) {
    const detail = (data as { detail?: unknown } | null)?.detail;
    const msg = res.status === 401 ? "This action needs the API key (set it under Data & Models)." : detailToText(detail, `Request failed (${res.status})`);
    throw new ApiError(res.status, msg);
  }
  return data as T;
}

export const api = {
  get: <T>(path: string) => request<T>("GET", path),
  post: <T>(path: string, body?: unknown) => request<T>("POST", path, body ?? {}),
  put: <T>(path: string, body: unknown) => request<T>("PUT", path, body),
  del: (path: string) => request<void>("DELETE", path),
  upload: <T>(path: string, file: File, fields: Record<string, string> = {}) => {
    const fd = new FormData();
    fd.append("file", file);
    for (const [k, v] of Object.entries(fields)) fd.append(k, v);
    return request<T>("POST", path, undefined, fd);
  },
};

export function wsUrl(): string {
  const explicit = import.meta.env.VITE_WS_URL as string | undefined;
  if (explicit) return explicit;
  if (BASE) return `${BASE.replace(/^http/, "ws")}/ws/stream`;
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws/stream`;
}

export function datasetFileUrl(name: string): string {
  return `${BASE}/api/v1/dataset/files/${encodeURIComponent(name)}`;
}

export function sampleUrl(name: string): string {
  return `${BASE}/api/v1/samples/${encodeURIComponent(name)}`;
}
