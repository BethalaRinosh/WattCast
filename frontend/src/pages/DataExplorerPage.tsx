import { useEffect, useState } from "react";
import { Download, Rows3 } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ApiError, getSeries, type SeriesResponse } from "../api/client";
import { EmptyState, ErrorState, LoadingState } from "./OverviewPage";

type Frequency = "hourly" | "daily";

function displayTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    ...(value.includes("T") && !/^\d{4}-\d{2}-\d{2}T00:00:00/.test(value) ? { hour: "2-digit" as const, minute: "2-digit" as const } : {}),
  });
}

export default function DataExplorerPage() {
  const [frequency, setFrequency] = useState<Frequency>("hourly");
  const [limit, setLimit] = useState(168);
  const [result, setResult] = useState<SeriesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    getSeries(frequency, limit, controller.signal)
      .then(setResult)
      .catch((cause: unknown) => {
        if (cause instanceof Error && cause.name === "AbortError") return;
        setResult(null);
        setError(cause instanceof ApiError || cause instanceof Error ? cause.message : "Unable to load the selected series.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [frequency, limit]);

  const points = result?.points ?? [];
  const chartData = points.map((point) => ({
    ...point,
    chartTime: new Date(point.timestamp).toLocaleString(undefined, frequency === "hourly"
      ? { month: "short", day: "numeric", hour: "2-digit" }
      : { month: "short", day: "numeric", year: "2-digit" }),
  }));
  const recentRows = [...points].reverse().slice(0, 20);
  const values = points.map((point) => point.active_power_kw).filter(Number.isFinite);
  const mean = values.length ? values.reduce((total, value) => total + value, 0) / values.length : null;
  const min = values.length ? Math.min(...values) : null;
  const max = values.length ? Math.max(...values) : null;

  function exportCsv() {
    if (!points.length) return;
    const escapeCsv = (value: string) => `"${value.replace(/"/g, '""')}"`;
    const lines = [
      "timestamp,active_power_kw",
      ...points.map((point) => `${escapeCsv(point.timestamp)},${point.active_power_kw}`),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `wattcast_${frequency}_series.csv`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-5">
      <section className="surface-card">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow">Query controls</p>
            <h2 className="mt-2 text-lg font-semibold tracking-tight">Choose the time resolution</h2>
            <p className="mt-1 max-w-xl text-sm leading-6 text-ink/50 dark:text-white/45">Inspect the latest cleaned observations served by the FastAPI series endpoint.</p>
          </div>
          <button type="button" onClick={exportCsv} disabled={!points.length || loading} className="inline-flex items-center gap-2 rounded-full border border-black/[0.1] px-4 py-2.5 text-xs font-medium transition hover:bg-black/[0.04] disabled:cursor-not-allowed disabled:opacity-40 dark:border-white/[0.12] dark:hover:bg-white/[0.05]">
            <Download size={14} /> Export CSV
          </button>
        </div>
        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          <label className="block text-xs font-medium text-ink/60 dark:text-white/55">
            Series frequency
            <select value={frequency} onChange={(event) => setFrequency(event.target.value as Frequency)} className="mt-2 block w-full rounded-xl border border-black/[0.1] bg-white/70 px-3 py-3 text-sm text-ink outline-none focus:border-lime-600 dark:border-white/[0.12] dark:bg-white/[0.04] dark:text-white">
              <option value="hourly">Hourly mean power</option>
              <option value="daily">Daily mean power</option>
            </select>
          </label>
          <label className="block text-xs font-medium text-ink/60 dark:text-white/55">
            Maximum observations
            <select value={limit} onChange={(event) => setLimit(Number(event.target.value))} className="mt-2 block w-full rounded-xl border border-black/[0.1] bg-white/70 px-3 py-3 text-sm text-ink outline-none focus:border-lime-600 dark:border-white/[0.12] dark:bg-white/[0.04] dark:text-white">
              {[24, 72, 168, 500, 1000, 2000].map((amount) => <option key={amount} value={amount}>{amount.toLocaleString()} points</option>)}
            </select>
          </label>
        </div>
      </section>

      {loading ? <LoadingState label={`Loading ${frequency} observations…`} /> : error ? <ErrorState title="Could not load this series" message={error} hint="Confirm preprocessing completed and the corresponding hourly_energy.csv or daily_energy.csv exists under backend/data/processed." /> : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <SmallStat label="Returned observations" value={(result?.count ?? 0).toLocaleString()} />
            <SmallStat label="Mean of selected window" value={mean === null ? "Unavailable" : `${mean.toFixed(3)} kW`} />
            <SmallStat label="Minimum in window" value={min === null ? "Unavailable" : `${min.toFixed(3)} kW`} />
            <SmallStat label="Maximum in window" value={max === null ? "Unavailable" : `${max.toFixed(3)} kW`} />
          </div>

          <section className="surface-card">
            <div className="mb-5 flex items-center justify-between gap-3">
              <div><p className="eyebrow">Time-series view</p><h2 className="mt-2 text-lg font-semibold tracking-tight">{frequency === "hourly" ? "Hourly" : "Daily"} mean active power</h2></div>
              <Rows3 size={18} className="text-ink/40 dark:text-white/40" />
            </div>
            {chartData.length ? <div className="h-[320px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                  <XAxis dataKey="chartTime" minTickGap={35} tick={{ fontSize: 11 }} stroke="currentColor" opacity={0.55} />
                  <YAxis width={58} tick={{ fontSize: 11 }} stroke="currentColor" opacity={0.55} tickFormatter={(value: number) => value.toFixed(1)} />
                  <Tooltip
                    contentStyle={{ borderRadius: 12, border: "1px solid #8884" }}
                    formatter={(value) => [typeof value === "number" ? `${value.toFixed(3)} kW` : value, "Mean active power"]}
                    labelFormatter={(_, payload) => payload?.[0]?.payload?.timestamp ? displayTimestamp(payload[0].payload.timestamp) : ""}
                  />
                  <Line type="monotone" dataKey="active_power_kw" stroke="#84b82f" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div> : <EmptyState message="No observations were returned for this selection." />}
          </section>

          <section className="surface-card">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
              <div><p className="eyebrow">Tabular data</p><h2 className="mt-2 text-lg font-semibold tracking-tight">Most recent observations</h2></div>
              <span className="text-xs text-ink/45 dark:text-white/40">Showing up to {Math.min(20, points.length)} of {points.length}</span>
            </div>
            {recentRows.length ? <div className="overflow-x-auto">
              <table className="w-full min-w-[420px] border-collapse text-left text-sm">
                <thead><tr className="border-b border-black/[0.08] text-xs text-ink/45 dark:border-white/[0.1] dark:text-white/40"><th className="py-3 pr-4 font-medium">Timestamp</th><th className="py-3 text-right font-medium">Mean active power (kW)</th></tr></thead>
                <tbody>{recentRows.map((point) => <tr key={point.timestamp} className="border-b border-black/[0.05] last:border-0 dark:border-white/[0.06]"><td className="py-3 pr-4 text-ink/65 dark:text-white/65">{displayTimestamp(point.timestamp)}</td><td className="py-3 text-right font-mono text-xs">{point.active_power_kw.toFixed(4)}</td></tr>)}</tbody>
              </table>
            </div> : <EmptyState message="No rows are available." />}
            <p className="mt-4 text-xs leading-5 text-ink/45 dark:text-white/40">The API returns the most recent observations in chronological order. CSV export downloads the currently selected response, not the entire original minute-level dataset.</p>
          </section>
        </>
      )}
    </div>
  );
}

function SmallStat({ label, value }: { label: string; value: string }) {
  return <section className="surface-card"><p className="eyebrow">{label}</p><p className="mt-3 break-words text-xl font-semibold tracking-tight">{value}</p></section>;
}
