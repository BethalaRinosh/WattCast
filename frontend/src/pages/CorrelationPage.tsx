import { useEffect, useMemo, useState } from "react";
import { ApiError, getCorrelation, type CorrelationResponse } from "../api/client";

type Frequency = "hourly" | "daily";
type Method = "pearson" | "spearman";

export default function CorrelationPage() {
  const [frequency, setFrequency] = useState<Frequency>("hourly");
  const [method, setMethod] = useState<Method>("pearson");
  const [data, setData] = useState<CorrelationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    getCorrelation(frequency, controller.signal)
      .then(setData)
      .catch((reason: unknown) => {
        if (reason instanceof Error && reason.name === "AbortError") return;
        setError(reason instanceof ApiError || reason instanceof Error ? reason.message : "Could not load correlation results.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [frequency]);

  const matrix = data?.[method] ?? {};
  const variables = data?.variables ?? [];
  const ranking = useMemo(() => data?.target_ranking ?? [], [data]);

  return (
    <div className="space-y-6">
      <section className="surface-card flex flex-wrap items-center justify-between gap-4">
        <div><p className="eyebrow">Association analysis</p><h2 className="mt-2 text-lg font-semibold">Correlation explorer</h2><p className="mt-1 text-sm text-ink/55 dark:text-white/50">Compare measurements against active power using two complementary correlation methods.</p></div>
        <div className="flex flex-wrap gap-2">
          <div className="inline-flex rounded-full border border-black/[0.08] p-1 dark:border-white/[0.1]">
            {(["hourly", "daily"] as const).map((value) => <button key={value} onClick={() => setFrequency(value)} className={`rounded-full px-3 py-2 text-xs capitalize ${frequency === value ? "bg-[#202a20] text-white dark:bg-white/15" : "text-ink/55 dark:text-white/50"}`}>{value}</button>)}
          </div>
          <div className="inline-flex rounded-full border border-black/[0.08] p-1 dark:border-white/[0.1]">
            {(["pearson", "spearman"] as const).map((value) => <button key={value} onClick={() => setMethod(value)} className={`rounded-full px-3 py-2 text-xs capitalize ${method === value ? "bg-[#202a20] text-white dark:bg-white/15" : "text-ink/55 dark:text-white/50"}`}>{value}</button>)}
          </div>
        </div>
      </section>

      {loading ? <section className="surface-card text-sm text-ink/55 dark:text-white/50">Loading correlation results…</section>
        : error ? <section className="surface-card" role="alert"><h3 className="font-semibold">Correlation results unavailable</h3><p className="mt-2 text-sm text-ink/55 dark:text-white/50">{error}</p><p className="mt-2 text-xs text-ink/40">Run the correlation analysis pipeline from the repository root.</p></section>
        : !data?.available ? <section className="surface-card"><p className="eyebrow">{data?.source_label ?? "Dataset"}</p><h3 className="mt-2 text-lg font-semibold">Not enough independent variables</h3><p className="mt-2 text-sm leading-6 text-ink/55 dark:text-white/50">{data?.message}</p><p className="mt-3 text-xs leading-5 text-ink/40">{data?.variables_note}</p></section>
        : <>
          <div className="grid gap-4 sm:grid-cols-3">
            <Stat label="Frequency" value={frequency} />
            <Stat label="Observations" value={data.observations.toLocaleString()} />
            <Stat label="Variables" value={String(variables.length)} />
          </div>
          <section className="surface-card">
            <div className="mb-5 flex flex-wrap items-end justify-between gap-3"><div><p className="eyebrow">{method} correlation</p><h3 className="mt-2 text-lg font-semibold">Measurement matrix</h3></div><span className="text-xs text-ink/40 dark:text-white/40">Scale: −1 to +1</span></div>
            <div className="overflow-x-auto">
              <table className="min-w-[620px] border-separate border-spacing-1 text-center text-xs">
                <thead><tr><th className="p-2" />{variables.map((name) => <th key={name} className="max-w-28 px-2 py-2 font-medium text-ink/55 dark:text-white/55">{name.replace(/_/g, " ")}</th>)}</tr></thead>
                <tbody>{variables.map((row) => <tr key={row}><th className="max-w-32 px-2 text-left font-medium text-ink/55 dark:text-white/55">{row.replace(/_/g, " ")}</th>{variables.map((column) => {
                  const value = matrix[row]?.[column] ?? null;
                  const alpha = value === null ? 0 : 0.12 + Math.abs(value) * 0.72;
                  const background = value === null ? "transparent" : value >= 0 ? `rgba(132, 190, 75, ${alpha})` : `rgba(220, 112, 105, ${alpha})`;
                  return <td key={column} title={value === null ? "Not enough paired observations" : `${row} / ${column}: ${value.toFixed(3)}`} className="min-w-16 rounded-lg px-2 py-3 font-mono" style={{ background }}>{value === null ? "N/A" : value.toFixed(2)}</td>;
                })}</tr>)}</tbody>
              </table>
            </div>
            <p className="mt-4 text-xs leading-5 text-ink/40 dark:text-white/40">{data.method}</p>
          </section>
          <section className="surface-card">
            <p className="eyebrow">Target relationship</p><h3 className="mt-2 text-lg font-semibold">Variables associated with active power</h3>
            {ranking.length ? <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[460px] text-left text-sm"><thead><tr className="border-b border-black/[0.08] text-xs text-ink/45 dark:border-white/[0.1] dark:text-white/40"><th className="py-3 pr-3 font-medium">Variable</th><th className="py-3 text-right font-medium">Pearson</th><th className="py-3 text-right font-medium">Spearman</th><th className="py-3 text-right font-medium">Paired rows</th></tr></thead><tbody>{ranking.map((item) => <tr key={item.variable} className="border-b border-black/[0.05] last:border-0 dark:border-white/[0.06]"><td className="py-3 pr-3">{item.variable.replace(/_/g, " ")}</td><td className="py-3 text-right font-mono">{item.pearson?.toFixed(3) ?? "N/A"}</td><td className="py-3 text-right font-mono">{item.spearman?.toFixed(3) ?? "N/A"}</td><td className="py-3 text-right font-mono">{item.paired_observations.toLocaleString()}</td></tr>)}</tbody></table></div> : <p className="mt-3 text-sm text-ink/50">No additional numeric variables are available.</p>}
          </section>
          <p className="px-1 text-xs leading-5 text-ink/45 dark:text-white/40">{data.interpretation} Correlation strength does not establish causation, and results can change with aggregation frequency.</p>
        </>}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return <section className="surface-card"><p className="eyebrow">{label}</p><p className="mt-3 text-xl font-semibold capitalize">{value}</p></section>;
}
