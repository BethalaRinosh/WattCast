const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();

export const API_BASE_URL = (configuredBaseUrl || "http://127.0.0.1:8000").replace(/\/+$/, "");

export interface HealthResponse {
  status: "ok";
  service: string;
  version: string;
}

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${normalizedPath}`, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(
      `Could not reach the WattCast API at ${API_BASE_URL}. Check that FastAPI is running and the URL is correct.`,
      0,
    );
  }

  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      typeof payload === "object" && payload !== null && "detail" in payload
        ? (payload as { detail: unknown }).detail
        : payload;
    const message =
      typeof detail === "object" && detail !== null && "message" in detail
        ? String((detail as { message: unknown }).message)
        : typeof detail === "string"
          ? detail
          : `API request failed with HTTP ${response.status}.`;
    throw new ApiError(message, response.status, detail);
  }

  return payload as T;
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return apiGet<HealthResponse>("/api/health", signal);
}
