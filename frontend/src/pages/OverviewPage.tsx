import { useEffect, useState } from "react";
import { Activity, CalendarDays, Gauge, TrendingUp } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ApiError, getSeries, getSummary, type SeriesResponse, type SummaryResponse } from "../api/client";
import { ChartZoomControls, useChartZoom } from "../components/ChartZoom";

function formatKw(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(3)} kW` : "Unavailable";
}

function formatDate(value: string | undefined): string {
  if (!value) return "Unavailable";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function readableModel(model: string | undefined): string {
  if (!model) return "Unavailable";
  const names: Record<string, string> = {
    naive: "Naive",
    seasonal_naive: "Seasonal naive",
    holt_winters: "Holt-Winters",
    arima_1_1_1: "ARIMA (1,1,1)",
    sarima: "SARIMA",
    random_forest_lag_calendar: "Random Forest",
  };
  return names[model] ?? model.replace(/_/g, " ");
}

export default function OverviewPage() {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [series, setSeries] = useState<SeriesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    Promise.all([getSummary(controller.signal), getSeries("hourly", 72, controller.signal)])
      .then(([summaryResult, seriesResult]) => {
        setSummary(summaryResult);
        setSeries(seriesResult);
      })
      .catch((cause: unknown) => {
        if (cause instanceof Error && cause.name === "AbortError") return;
        setError(cause instanceof ApiError || cause instanceof Error ? cause.message : "Unable to load WattCast overview.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  const hourly = summary?.hourly;
  const daily = summary?.daily;
  const hourlyEvaluation = summary?.evaluation?.series?.hourly;
  const sourceLabel = summary?.source_label ?? "Source not specified";

  const chartData = (series?.points ?? []).map((point, chartIndex) => ({
    ...point,
    chartIndex,
    chartTime: new Date(point.timestamp).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit" }),
  }));
  const zoom = useChartZoom(chartData.length);

  if (loading) return <LoadingState label="Loading summary and recent hourly observations…" />;
  if (error) return <ErrorState title="Overview data could not be loaded" message={error} hint="Check that the backend is running and that Steps 3, 4, and 9 have generated their saved artifacts." />;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="eyebrow">Dataset provenance</p>
          <h2 className="mt-1 text-sm font-medium">{sourceLabel}</h2>
        </div>
        <span className="rounded-full border border-black/[0.08] px-3 py-1.5 text-xs text-ink/55 dark:border-white/[0.1] dark:text-white/55">
          Target: mean active power (kW)
        </span>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard icon={Gauge} label="Hourly average" value={formatKw(hourly?.mean_kw)} detail={hourly ? `${hourly.count.toLocaleString()} hourly observations` : "No hourly summary"} />
        <MetricCard icon={TrendingUp} label="Hourly peak" value={formatKw(hourly?.max_kw)} detail={hourly ? `Minimum ${formatKw(hourly.min_kw)}` : "No hourly summary"} />
        <MetricCard icon={CalendarDays} label="Daily average" value={formatKw(daily?.mean_kw)} detail={daily ? `${daily.count.toLocaleString()} daily observations` : "No daily summary"} />
        <MetricCard icon={Activity} label="Best hourly MAE" value={readableModel(hourlyEvaluation?.best_by_mae)} detail={hourlyEvaluation?.evaluation_observations ? `${hourlyEvaluation.evaluation_observations.toLocaleString()} holdout observations` : "Evaluation details unavailable"} />
      </div>

      <section className="surface-card">
        <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="eyebrow">Recent observations</p>
            <h2 className="mt-2 text-lg font-semibold tracking-tight">Hourly electricity use</h2>
            <p className="mt-1 text-sm text-ink/50 dark:text-white/45">Latest {series?.count ?? 0} observations returned by the API.</p>
          </div>
          <div className="flex flex-col items-end gap-2 text-right text-xs text-ink/45 dark:text-white/40">
            <p>From {formatDate(hourly?.start)}</p>
            <p>To {formatDate(hourly?.end)}</p>
            <ChartZoomControls isZoomed={zoom.isZoomed} onReset={zoom.resetZoom} />
          </div>
        </div>
        {chartData.length ? (
          <div className="h-[300px] w-full" ref={zoom.containerRef} style={{ overscrollBehavior: "contain" }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                <XAxis dataKey="chartIndex" type="number" domain={zoom.domain} allowDataOverflow tickFormatter={(value: number) => chartData[Math.round(value)]?.chartTime ?? ""} minTickGap={36} tick={{ fontSize: 11 }} stroke="currentColor" opacity={0.55} />
                <YAxis width={58} tick={{ fontSize: 11 }} stroke="currentColor" opacity={0.55} tickFormatter={(value: number) => value.toFixed(1)} />
                <Tooltip
                  contentStyle={{ borderRadius: 12, border: "1px solid #8884", background: "var(--tooltip-bg, #ffffff)" }}
                  formatter={(value) => [typeof value === "number" ? `${value.toFixed(3)} kW` : value, "Mean active power"]}
                  labelFormatter={(_, payload) => payload?.[0]?.payload?.timestamp ? new Date(payload[0].payload.timestamp).toLocaleString() : ""}
                />
                <Line type="monotone" dataKey="active_power_kw" stroke="#84b82f" strokeWidth={2} dot={false} activeDot={{ r: 4 }} name="Mean active power" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : <EmptyState message="The API returned no hourly observations." />}
        <p className="mt-3 text-xs leading-5 text-ink/45 dark:text-white/40">The chart displays saved observations, not a future forecast. Summary statistics cover the full processed series; the chart shows only the recent window requested from the API.</p>
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <InfoCard title="Hourly coverage" value={hourly ? `${formatDate(hourly.start)} – ${formatDate(hourly.end)}` : "Unavailable"} detail={hourly ? `Median ${formatKw(hourly.median_kw)} · standard deviation ${formatKw(hourly.std_kw)}` : "The EDA summary did not include hourly statistics."} />
        <InfoCard title="Daily coverage" value={daily ? `${formatDate(daily.start)} – ${formatDate(daily.end)}` : "Unavailable"} detail={daily ? `Median ${formatKw(daily.median_kw)} · standard deviation ${formatKw(daily.std_kw)}` : "The EDA summary did not include daily statistics."} />
      </div>
    </div>
  );
}

function MetricCard({ icon: Icon, label, value, detail }: { icon: LucideIcon; label: string; value: string; detail: string }) {
  return (
    <section className="surface-card min-w-0">
      <div className="flex items-center justify-between gap-2">
        <p className="eyebrow">{label}</p>
        <Icon size={17} className="text-ink/40 dark:text-white/40" />
      </div>
      <p className="mt-4 break-words text-2xl font-semibold tracking-tight">{value}</p>
      <p className="mt-2 text-xs leading-5 text-ink/45 dark:text-white/40">{detail}</p>
    </section>
  );
}

function InfoCard({ title, value, detail }: { title: string; value: string; detail: string }) {
  return <section className="surface-card"><p className="eyebrow">{title}</p><p className="mt-3 text-sm font-medium leading-6">{value}</p><p className="mt-2 text-xs leading-5 text-ink/45 dark:text-white/40">{detail}</p></section>;
}

export function LoadingState({ label }: { label: string }) {
  return <div className="surface-card flex min-h-48 items-center gap-3 text-sm text-ink/55 dark:text-white/50"><span className="h-4 w-4 animate-spin rounded-full border-2 border-lime-500 border-t-transparent" />{label}</div>;
}

export function ErrorState({ title, message, hint }: { title: string; message: string; hint: string }) {
  return <div className="rounded-2xl border border-rose-500/25 bg-rose-500/[0.06] p-5 sm:p-6" role="alert"><h2 className="font-medium">{title}</h2><p className="mt-2 break-words text-sm leading-6 text-ink/60 dark:text-white/55">{message}</p><p className="mt-3 text-xs leading-5 text-ink/45 dark:text-white/40">{hint}</p></div>;
}

export function EmptyState({ message }: { message: string }) {
  return <div className="flex h-48 items-center justify-center rounded-xl border border-dashed border-black/10 px-4 text-center text-sm text-ink/45 dark:border-white/10 dark:text-white/40">{message}</div>;
}
