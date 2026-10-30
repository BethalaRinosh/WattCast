const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();

export const API_BASE_URL = (configuredBaseUrl || "http://127.0.0.1:8000").replace(/\/+$/, "");

export interface HealthResponse {
  status: "ok";
  service: string;
  version: string;
}

export interface SeriesPoint {
  timestamp: string;
  active_power_kw: number;
}

export interface SeriesResponse {
  series: "hourly" | "daily";
  count: number;
  points: SeriesPoint[];
}

export interface DescriptiveStats {
  count: number;
  mean_kw: number;
  std_kw: number | null;
  min_kw: number;
  q25_kw: number;
  median_kw: number;
  q75_kw: number;
  max_kw: number;
  start: string;
  end: string;
}

export interface ModelMetrics {
  mae?: number;
  rmse?: number;
  mape_percent?: number;
}


export interface ResidualDiagnostics {
  mean_error_kw?: number;
  median_error_kw?: number;
  residual_std_kw?: number;
  mean_absolute_error_kw?: number;
  max_absolute_error_kw?: number;
  lag_1_residual_autocorrelation?: number | null;
  fraction_within_two_residual_std?: number;
  interpretation_note?: string;
}

export interface EvaluatedModel {
  status?: string;
  metrics?: ModelMetrics;
  residual_diagnostics?: ResidualDiagnostics;
  evaluated_timestamps?: number;
  evaluation_start?: string;
  evaluation_end?: string;
}

export interface SeriesEvaluation {
  series?: "hourly" | "daily";
  best_by_mae?: string;
  best_by_rmse?: string;
  evaluation_observations?: number;
  ranking_by_mae_then_rmse?: Array<{ model: string; mae: number; rmse: number }>;
  models?: Record<string, EvaluatedModel>;
}

export interface ModelComparisonResponse extends SeriesEvaluation {
  series: "hourly" | "daily";
}

export interface ForecastPoint {
  timestamp: string;
  actual_kw: number;
  predicted_kw: number;
}

export interface ForecastResponse {
  series: "hourly" | "daily";
  model: string;
  count: number;
  points: ForecastPoint[];
  note: string;
}

export interface SummaryResponse {
  source_type: "real" | "synthetic" | "unknown" | null;
  source_label: string | null;
  hourly: DescriptiveStats | null;
  daily: DescriptiveStats | null;
  evaluation: {
    series?: Record<string, SeriesEvaluation>;
    [key: string]: unknown;
  } | null;
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

export function getSummary(signal?: AbortSignal): Promise<SummaryResponse> {
  return apiGet<SummaryResponse>("/api/summary", signal);
}

export function getSeries(
  series: "hourly" | "daily",
  limit = 500,
  signal?: AbortSignal,
): Promise<SeriesResponse> {
  const safeLimit = Math.min(2000, Math.max(1, Math.floor(limit)));
  return apiGet<SeriesResponse>(`/api/series/${series}?limit=${safeLimit}`, signal);
}

export interface CorrelationVariable {
  variable: string;
  pearson: number | null;
  spearman: number | null;
  paired_observations: number;
}

export interface CorrelationResponse {
  project: string;
  source_type: "real" | "synthetic" | "unknown";
  source_label: string;
  variables_note?: string;
  series: "hourly" | "daily";
  available: boolean;
  message?: string;
  method?: string;
  observations: number;
  variables: string[];
  pearson?: Record<string, Record<string, number | null>>;
  spearman?: Record<string, Record<string, number | null>>;
  target_ranking?: CorrelationVariable[];
  interpretation?: string;
}

export function getCorrelation(
  series: "hourly" | "daily",
  signal?: AbortSignal,
): Promise<CorrelationResponse> {
  return apiGet<CorrelationResponse>(`/api/correlation/${series}`, signal);
}

export interface DecompositionPoint {
  timestamp: string;
  observed: number | null;
  trend: number | null;
  seasonal: number | null;
  residual: number | null;
}

export interface DecompositionResponse {
  series: "hourly" | "daily";
  count: number;
  points: DecompositionPoint[];
}

export function getDecomposition(
  series: "hourly" | "daily",
  signal?: AbortSignal,
): Promise<DecompositionResponse> {
  return apiGet<DecompositionResponse>(`/api/decomposition/${series}`, signal);
}

export function getModels(
  series: "hourly" | "daily",
  signal?: AbortSignal,
): Promise<ModelComparisonResponse> {
  return apiGet<ModelComparisonResponse>(`/api/models/${series}`, signal);
}

export function getForecast(
  series: "hourly" | "daily",
  model: string,
  limit = 500,
  signal?: AbortSignal,
): Promise<ForecastResponse> {
  const safeLimit = Math.min(2000, Math.max(1, Math.floor(limit)));
  const params = new URLSearchParams({ model, limit: String(safeLimit) });
  return apiGet<ForecastResponse>(`/api/forecast/${series}?${params.toString()}`, signal);
}
