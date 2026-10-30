import { useCallback, useEffect, useState, type WheelEvent } from "react";

type ChartZoom = {
  domain: [number, number];
  handleWheel: (event: WheelEvent<HTMLDivElement>) => void;
  resetZoom: () => void;
  isZoomed: boolean;
};

export function useChartZoom(pointCount: number, resetKey: string | number = pointCount): ChartZoom {
  const lastIndex = Math.max(0, pointCount - 1);
  const [domain, setDomain] = useState<[number, number]>([0, lastIndex]);

  useEffect(() => {
    setDomain([0, lastIndex]);
  }, [lastIndex, resetKey]);

  const resetZoom = useCallback(() => setDomain([0, lastIndex]), [lastIndex]);

  const handleWheel = useCallback((event: WheelEvent<HTMLDivElement>) => {
    if (pointCount < 2) return;
    event.preventDefault();

    const [start, end] = domain;
    const currentSpan = Math.max(1, end - start);
    const minimumSpan = Math.min(10, lastIndex);
    const factor = event.deltaY < 0 ? 0.8 : 1.25;
    const nextSpan = Math.min(lastIndex, Math.max(minimumSpan, currentSpan * factor));
    if (Math.abs(nextSpan - currentSpan) < 0.01) return;

    const bounds = event.currentTarget.getBoundingClientRect();
    const pointerRatio = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width));
    const anchor = start + currentSpan * pointerRatio;
    let nextStart = anchor - nextSpan * pointerRatio;
    let nextEnd = nextStart + nextSpan;

    if (nextStart < 0) {
      nextEnd -= nextStart;
      nextStart = 0;
    }
    if (nextEnd > lastIndex) {
      nextStart -= nextEnd - lastIndex;
      nextEnd = lastIndex;
    }

    setDomain([Math.max(0, nextStart), Math.min(lastIndex, nextEnd)]);
  }, [domain, lastIndex, pointCount]);

  return {
    domain,
    handleWheel,
    resetZoom,
    isZoomed: domain[0] > 0.5 || domain[1] < lastIndex - 0.5,
  };
}

export function ChartZoomControls({ isZoomed, onReset }: { isZoomed: boolean; onReset: () => void }) {
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs text-ink/45 dark:text-white/40">
      <span>Scroll over chart to zoom</span>
      {isZoomed && (
        <button type="button" onClick={onReset} className="rounded-full border border-black/10 px-3 py-1.5 font-medium text-ink/65 transition hover:bg-black/[0.04] dark:border-white/15 dark:text-white/65 dark:hover:bg-white/[0.06]">
          Reset zoom
        </button>
      )}
    </div>
  );
}
