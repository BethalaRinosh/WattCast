import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, getModels, type ModelComparisonResponse } from "../api/client";
import { ErrorState, LoadingState } from "./OverviewPage";

type Frequency = "hourly" | "daily";
type Metric = "mae" | "rmse" | "mape";

const MODEL_NAMES: Record<string, string> = {
  naive: "Naive",
  seasonal_naive: "Seasonal naive",
  holt_winters: "Holt-Winters",
  arima_1_1_1: "ARIMA (1,1,1)",
  sarima: "SARIMA",
  random_forest_lag_calendar: "Random Forest",
};

const METRIC_LABELS: Record<Metric, string> = {
  mae: "MAE (kW)",
  rmse: "RMSE (kW)",
  mape: "MAPE (%)",
};

function modelName(value: string): string {
  return MODEL_NAMES[value] ?? value.replace(/_/g, " ");
}

function metricValue(value: number | undefined, metric: Metric): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  return metric === "mape" ? `${value.toFixed(2)}%` : `${value.toFixed(4)} kW`;
}

export default function ModelLabPage() {
  const [frequency, setFrequency] = useState<Frequency>("hourly");
  const [metric, setMetric] = useState<Metric>("mae");
  const [result, setResult] = useState<ModelComparisonResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    getModels(frequency, controller.signal)
      .then(setResult)
      .catch((cause: unknown) => {
        if (cause instanceof Error && cause.name === "AbortError") return;
        setResult(null);
        setError(cause instanceof ApiError || cause instanceof Error ? cause.message : "Unable to load model evaluation.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [frequency]);

  const models = result?.models ?? {};
  const rows = useMemo(() => Object.entries(models)
    .filter(([, model]) => model.status === "success" && typeof model.metrics?.[metric] === "number")
    .map(([key, model]) => ({ key, name: modelName(key), score: model.metrics?.[metric] as number }))
    .sort((a, b) => a.score - b.score), [models, metric]);
  const bestModel = metric === "rmse" ? result?.best_by_rmse : result?.best_by_mae;
  const bestScore = bestModel ? models[bestModel]?.metrics?.[metric] : undefined;

  return (
    <div className="space-y-5">
      <section className="surface-card flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow">Holdout evaluation</p>
          <h2 className="mt-2 text-lg font-semibold">Compare forecasting models</h2>
          <p className="mt-1 max-w-xl text-sm leading-6 text-ink/50 dark:text-white/45">Every candidate is scored on aligned timestamps from the same chronological test period. Lower error is better.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <select aria-label="Series frequency" value={frequency} onChange={(event) => setFrequency(event.target.value as Frequency)} className="rounded-xl border border-black/10 bg-white/70 px-3 py-2.5 text-xs dark:border-white/10 dark:bg-white/[0.04] dark:text-white">
            <option value="hourly">Hourly</option><option value="daily">Daily</option>
          </select>
          <select aria-label="Evaluation metric" value={metric} onChange={(event) => setMetric(event.target.value as Metric)} className="rounded-xl border border-black/10 bg-white/70 px-3 py-2.5 text-xs dark:border-white/10 dark:bg-white/[0.04] dark:text-white">
            <option value="mae">MAE</option><option value="rmse">RMSE</option><option value="mape">MAPE</option>
          </select>
        </div>
      </section>

      {loading ? <LoadingState label={`Loading ${frequency} model evaluation…`} />
        : error ? <ErrorState title="Model evaluation could not be loaded" message={error} hint="Run scripts/model_evaluation.py and confirm the evaluation_results.json artifact exists." />
        : <>
          <div className="grid gap-4 sm:grid-cols-3">
            <Stat label="Frequency" value={frequency} />
            <Stat label="Shared evaluation observations" value={(result?.evaluation_observations ?? 0).toLocaleString()} />
            <Stat label={`Best by ${metric.toUpperCase()}`} value={bestModel ? modelName(bestModel) : "Unavailable"} />
          </div>
          {bestModel && <section className="surface-card">
            <p className="eyebrow">Lowest {metric.toUpperCase()}</p>
            <div className="mt-2 flex flex-wrap items-baseline justify-between gap-3">
              <h3 className="text-2xl font-semibold">{modelName(bestModel)}</h3>
              <p className="font-mono text-xl">{metricValue(bestScore, metric)}</p>
            </div>
            <p className="mt-2 text-xs leading-5 text-ink/45 dark:text-white/40">Best-model labels are selected from the saved evaluation results. MAPE is useful as a percentage summary but can be unstable around near-zero actual values.</p>
          </section>}
          <section className="surface-card">
            <div className="mb-5"><p className="eyebrow">{METRIC_LABELS[metric]}</p><h3 className="mt-2 text-lg font-semibold">Model error ranking</h3></div>
            {rows.length ? <div className="h-[320px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 12 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="currentColor" opacity={0.1} />
                  <XAxis type="number" tick={{ fontSize: 11 }} stroke="currentColor" opacity={0.55} />
                  <YAxis type="category" dataKey="name" width={112} tick={{ fontSize: 11 }} stroke="currentColor" opacity={0.65} />
                  <Tooltip formatter={(value) => [metricValue(Number(value), metric), METRIC_LABELS[metric]]} />
                  <Bar dataKey="score" fill="#84b82f" radius={[0, 6, 6, 0]} maxBarSize={28} />
                </BarChart>
              </ResponsiveContainer>
            </div> : <p className="text-sm text-ink/50">No successful model metrics are available for this selection.</p>}
          </section>
          <section className="surface-card">
            <p className="eyebrow">Detailed results</p><h3 className="mt-2 text-lg font-semibold">All evaluated candidates</h3>
            <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[560px] text-left text-sm">
              <thead><tr className="border-b border-black/[0.08] text-xs text-ink/45 dark:border-white/10"><th className="py-3 pr-3 font-medium">Model</th><th className="py-3 text-right font-medium">MAE (kW)</th><th className="py-3 text-right font-medium">RMSE (kW)</th><th className="py-3 text-right font-medium">MAPE (%)</th><th className="py-3 text-right font-medium">Rows</th></tr></thead>
              <tbody>{Object.entries(models).map(([key, model]) => <tr key={key} className="border-b border-black/[0.05] last:border-0 dark:border-white/[0.06]">
                <td className="py-3 pr-3 font-medium">{modelName(key)}</td>
                <td className="py-3 text-right font-mono text-xs">{metricValue(model.metrics?.mae, "mae")}</td>
                <td className="py-3 text-right font-mono text-xs">{metricValue(model.metrics?.rmse, "rmse")}</td>
                <td className="py-3 text-right font-mono text-xs">{metricValue(model.metrics?.mape, "mape")}</td>
                <td className="py-3 text-right font-mono text-xs">{model.evaluated_timestamps?.toLocaleString() ?? "N/A"}</td>
              </tr>)}</tbody>
            </table></div>
            <p className="mt-4 text-xs leading-5 text-ink/45 dark:text-white/40">MAE is average absolute error. RMSE penalises larger errors more strongly. MAPE expresses percentage error and excludes zero or near-zero actual values according to the evaluation pipeline.</p>
          </section>
        </>}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return <section className="surface-card"><p className="eyebrow">{label}</p><p className="mt-3 break-words text-xl font-semibold capitalize">{value}</p></section>;
}
