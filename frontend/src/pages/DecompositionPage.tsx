import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, getDecomposition, type DecompositionResponse } from "../api/client";
import { EmptyState, ErrorState, LoadingState } from "./OverviewPage";
import { ChartZoomControls, useChartZoom } from "../components/ChartZoom";

type Frequency = "hourly" | "daily";
type ComponentKey = "observed" | "trend" | "seasonal" | "residual";

const COMPONENTS: Array<{ key: ComponentKey; title: string; description: string }> = [
  { key: "observed", title: "Observed series", description: "The original mean active-power measurements." },
  { key: "trend", title: "Trend", description: "The estimated longer-term movement in the series." },
  { key: "seasonal", title: "Seasonal component", description: "The repeating pattern estimated by additive decomposition." },
  { key: "residual", title: "Residual", description: "What remains after subtracting trend and seasonality." },
];

function formatTimestamp(value: string, frequency: Frequency): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, frequency === "hourly"
    ? { month: "short", day: "numeric", hour: "2-digit" }
    : { month: "short", day: "numeric", year: "2-digit" });
}

export default function DecompositionPage() {
  const [frequency, setFrequency] = useState<Frequency>("hourly");
  const [data, setData] = useState<DecompositionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    getDecomposition(frequency, controller.signal)
      .then(setData)
      .catch((cause: unknown) => {
        if (cause instanceof Error && cause.name === "AbortError") return;
        setData(null);
        setError(cause instanceof ApiError || cause instanceof Error ? cause.message : "Unable to load decomposition components.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [frequency]);

  const chartData = useMemo(() => (data?.points ?? []).map((point, chartIndex) => ({
    ...point,
    chartIndex,
    chartTime: formatTimestamp(point.timestamp, frequency),
  })), [data, frequency]);
  const zoom = useChartZoom(chartData.length, frequency);

  return (
    <div className="space-y-5">
      <section className="surface-card flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow">Classical time-series decomposition</p>
          <h2 className="mt-2 text-lg font-semibold">Separate the parts of energy use</h2>
          <p className="mt-1 max-w-2xl text-sm leading-6 text-ink/50 dark:text-white/45">Additive decomposition represents the observed series as trend + seasonal component + residual. It is a diagnostic view, not a forecasting model.</p>
        </div>
        <label className="block min-w-40 text-xs font-medium text-ink/60 dark:text-white/55">Time resolution
          <select value={frequency} onChange={(event) => setFrequency(event.target.value as Frequency)} className="mt-2 block w-full rounded-xl border border-black/10 bg-white/70 px-3 py-3 text-sm dark:border-white/10 dark:bg-white/[0.04] dark:text-white">
            <option value="hourly">Hourly · period 24</option>
            <option value="daily">Daily · period 7</option>
          </select>
        </label>
      </section>

      {loading ? <LoadingState label={`Loading ${frequency} decomposition…`} />
        : error ? <ErrorState title="Decomposition could not be loaded" message={error} hint="Run python scripts/time_series_analysis.py from the repository root and confirm its CSV outputs exist." />
        : !chartData.length ? <EmptyState message="No decomposition rows are available for this frequency." />
        : <>
          <div className="grid gap-4 sm:grid-cols-3">
            <Stat label="Frequency" value={frequency} />
            <Stat label="Returned rows" value={chartData.length.toLocaleString()} />
            <Stat label="Seasonal period" value={frequency === "hourly" ? "24 observations" : "7 observations"} />
          </div>
          <section className="surface-card">
            <p className="eyebrow">Decomposition model</p>
            <h3 className="mt-2 text-lg font-semibold">Additive components</h3>
            <p className="mt-2 text-sm leading-6 text-ink/55 dark:text-white/50">Observed = Trend + Seasonal + Residual. The hourly analysis uses a 24-point daily cycle; the daily analysis uses a 7-point weekly cycle. The pipeline regularizes timestamps and uses its longest contiguous segment rather than bridging long data gaps.</p>
          </section>
          <div className="grid gap-4 xl:grid-cols-2">
            {COMPONENTS.map((component) => {
              const validCount = chartData.filter((point) => typeof point[component.key] === "number").length;
              return <section className="surface-card min-w-0" key={component.key}>
                <div className="mb-4 flex flex-wrap items-start justify-between gap-2"><div><p className="eyebrow">{component.title}</p><p className="mt-2 text-xs leading-5 text-ink/45 dark:text-white/40">{component.description}</p></div><ChartZoomControls isZoomed={zoom.isZoomed} onReset={zoom.resetZoom} /></div>
                {validCount ? <div className="h-[250px] w-full" ref={zoom.containerRef} style={{ overscrollBehavior: "contain" }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                      <XAxis dataKey="chartIndex" type="number" domain={zoom.domain} allowDataOverflow tickFormatter={(value: number) => chartData[Math.round(value)]?.chartTime ?? ""} minTickGap={32} tick={{ fontSize: 10 }} stroke="currentColor" opacity={0.5} />
                      <YAxis width={58} tick={{ fontSize: 10 }} stroke="currentColor" opacity={0.5} tickFormatter={(value: number) => value.toFixed(1)} />
                      <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #8884" }} formatter={(value) => [typeof value === "number" ? `${value.toFixed(4)} kW` : value, component.title]} labelFormatter={(_, payload) => payload?.[0]?.payload?.timestamp ? new Date(payload[0].payload.timestamp).toLocaleString() : ""} />
                      <Line type="monotone" dataKey={component.key} name={component.title} stroke="#84b82f" strokeWidth={1.7} dot={false} connectNulls={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div> : <EmptyState message={`No valid ${component.title.toLowerCase()} values are available.`} />}
                <p className="mt-2 text-xs text-ink/40 dark:text-white/35">{validCount.toLocaleString()} non-missing values</p>
              </section>;
            })}
          </div>
          <p className="px-1 text-xs leading-5 text-ink/40 dark:text-white/35">Additive decomposition assumes the seasonal effect is approximately constant in magnitude. Edge values or missing component values can occur because trend estimation uses neighboring observations.</p>
        </>}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return <section className="surface-card"><p className="eyebrow">{label}</p><p className="mt-3 text-xl font-semibold capitalize">{value}</p></section>;
}
