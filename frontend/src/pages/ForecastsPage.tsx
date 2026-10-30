import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, getForecast, type ForecastResponse } from "../api/client";
import { ErrorState, LoadingState } from "./OverviewPage";

type Frequency = "hourly" | "daily";
const MODELS = [
  ["naive", "Naive"],
  ["seasonal_naive", "Seasonal naive"],
  ["holt_winters", "Holt-Winters"],
  ["arima_1_1_1", "ARIMA (1,1,1)"],
  ["sarima", "SARIMA"],
  ["random_forest_lag_calendar", "Random Forest"],
] as const;

function timestampLabel(value: string, frequency: Frequency): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, frequency === "hourly"
    ? { month: "short", day: "numeric", hour: "2-digit" }
    : { month: "short", day: "numeric", year: "2-digit" });
}

export default function ForecastsPage() {
  const [frequency, setFrequency] = useState<Frequency>("hourly");
  const [model, setModel] = useState<string>("holt_winters");
  const [limit, setLimit] = useState(500);
  const [result, setResult] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    getForecast(frequency, model, limit, controller.signal)
      .then(setResult)
      .catch((cause: unknown) => {
        if (cause instanceof Error && cause.name === "AbortError") return;
        setResult(null);
        setError(cause instanceof ApiError || cause instanceof Error ? cause.message : "Unable to load saved forecast.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [frequency, model, limit]);

  const points = result?.points ?? [];
  const chartData = useMemo(() => points.map((point) => ({
    ...point,
    chartTime: timestampLabel(point.timestamp, frequency),
  })), [points, frequency]);
  const mae = points.length ? points.reduce((sum, point) => sum + Math.abs(point.actual_kw - point.predicted_kw), 0) / points.length : null;
  const rmse = points.length ? Math.sqrt(points.reduce((sum, point) => sum + (point.actual_kw - point.predicted_kw) ** 2, 0) / points.length) : null;

  return (
    <div className="space-y-5">
      <section className="surface-card">
        <p className="eyebrow">Forecast inspection</p><h2 className="mt-2 text-lg font-semibold">Actual versus predicted power</h2>
        <p className="mt-1 max-w-2xl text-sm leading-6 text-ink/50 dark:text-white/45">Inspect a saved model's predictions over the chronological holdout period. These are historical test predictions, not forecasts beyond the dataset.</p>
        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          <label className="block text-xs font-medium text-ink/60 dark:text-white/55">Frequency
            <select value={frequency} onChange={(event) => setFrequency(event.target.value as Frequency)} className="mt-2 block w-full rounded-xl border border-black/10 bg-white/70 px-3 py-3 text-sm dark:border-white/10 dark:bg-white/[0.04] dark:text-white"><option value="hourly">Hourly</option><option value="daily">Daily</option></select>
          </label>
          <label className="block text-xs font-medium text-ink/60 dark:text-white/55">Forecast model
            <select value={model} onChange={(event) => setModel(event.target.value)} className="mt-2 block w-full rounded-xl border border-black/10 bg-white/70 px-3 py-3 text-sm dark:border-white/10 dark:bg-white/[0.04] dark:text-white">{MODELS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
          </label>
          <label className="block text-xs font-medium text-ink/60 dark:text-white/55">Holdout points
            <select value={limit} onChange={(event) => setLimit(Number(event.target.value))} className="mt-2 block w-full rounded-xl border border-black/10 bg-white/70 px-3 py-3 text-sm dark:border-white/10 dark:bg-white/[0.04] dark:text-white">{[100, 250, 500, 1000, 2000].map((value) => <option key={value} value={value}>{value.toLocaleString()} points</option>)}</select>
          </label>
        </div>
      </section>

      {loading ? <LoadingState label="Loading saved holdout predictions…" />
        : error ? <ErrorState title="Forecast could not be loaded" message={error} hint="Confirm Steps 6–9 generated the forecast CSV for this frequency and model." />
        : <>
          <div className="grid gap-4 sm:grid-cols-3">
            <Stat label="Returned holdout points" value={points.length.toLocaleString()} />
            <Stat label="MAE in selected window" value={mae === null ? "N/A" : `${mae.toFixed(4)} kW`} />
            <Stat label="RMSE in selected window" value={rmse === null ? "N/A" : `${rmse.toFixed(4)} kW`} />
          </div>
          <section className="surface-card">
            <div className="mb-5 flex flex-wrap items-end justify-between gap-3"><div><p className="eyebrow">Chronological holdout</p><h3 className="mt-2 text-lg font-semibold">Observed and modelled values</h3></div><span className="text-xs text-ink/40">{frequency} · {model.replace(/_/g, " ")}</span></div>
            {chartData.length ? <div className="h-[360px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                  <XAxis dataKey="chartTime" minTickGap={35} tick={{ fontSize: 11 }} stroke="currentColor" opacity={0.55} />
                  <YAxis width={62} tick={{ fontSize: 11 }} stroke="currentColor" opacity={0.55} tickFormatter={(value: number) => value.toFixed(1)} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #8884" }} formatter={(value) => [typeof value === "number" ? `${value.toFixed(3)} kW` : value]} labelFormatter={(_, payload) => payload?.[0]?.payload?.timestamp ? new Date(payload[0].payload.timestamp).toLocaleString() : ""} />
                  <Legend />
                  <Line type="monotone" dataKey="actual_kw" name="Actual" stroke="#84b82f" strokeWidth={2} dot={false} activeDot={{ r: 3 }} />
                  <Line type="monotone" dataKey="predicted_kw" name="Predicted" stroke="#6f83d9" strokeWidth={2} dot={false} activeDot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div> : <p className="text-sm text-ink/50">No prediction points were returned.</p>}
            <p className="mt-4 text-xs leading-5 text-ink/45 dark:text-white/40">{result?.note} The displayed MAE/RMSE are recalculated for the selected returned window; use Model lab for the full shared evaluation comparison.</p>
          </section>
          <section className="surface-card">
            <p className="eyebrow">Prediction sample</p><h3 className="mt-2 text-lg font-semibold">Latest returned holdout rows</h3>
            <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[520px] text-left text-sm"><thead><tr className="border-b border-black/[0.08] text-xs text-ink/45 dark:border-white/10"><th className="py-3 pr-3">Timestamp</th><th className="py-3 text-right">Actual (kW)</th><th className="py-3 text-right">Predicted (kW)</th><th className="py-3 text-right">Error (kW)</th></tr></thead><tbody>{points.slice(-12).reverse().map((point) => <tr key={point.timestamp} className="border-b border-black/[0.05] last:border-0 dark:border-white/[0.06]"><td className="py-3 pr-3 text-xs">{timestampLabel(point.timestamp, frequency)}</td><td className="py-3 text-right font-mono text-xs">{point.actual_kw.toFixed(3)}</td><td className="py-3 text-right font-mono text-xs">{point.predicted_kw.toFixed(3)}</td><td className="py-3 text-right font-mono text-xs">{(point.actual_kw - point.predicted_kw).toFixed(3)}</td></tr>)}</tbody></table></div>
          </section>
        </>}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return <section className="surface-card"><p className="eyebrow">{label}</p><p className="mt-3 break-words text-xl font-semibold tracking-tight">{value}</p></section>;
}
