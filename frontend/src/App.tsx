import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  BarChart3,
  ChevronRight,
  CircleHelp,
  Command,
  Database,
  GitBranch,
  Gauge,
  LayoutDashboard,
  Moon,
  Sun,
  Waves,
  Zap,
} from "lucide-react";
import { ApiError, getHealth } from "./api/client";
import OverviewPage from "./pages/OverviewPage";
import DataExplorerPage from "./pages/DataExplorerPage";
import CorrelationPage from "./pages/CorrelationPage";

const navigation = [
  { label: "Overview", icon: LayoutDashboard, description: "A clear view of your energy data." },
  { label: "Data explorer", icon: Database, description: "Inspect the hourly and daily series." },
  { label: "Correlation analysis", icon: GitBranch, description: "Measure relationships between household power variables." },
  { label: "Decomposition", icon: Waves, description: "Separate trend, seasonality and residuals." },
  { label: "Model lab", icon: BarChart3, description: "Compare forecasting approaches." },
  { label: "Forecasts", icon: Gauge, description: "Review predictions against held-out data." },
] as const;

type PageName = (typeof navigation)[number]["label"];
type ConnectionState = "checking" | "connected" | "error";

export default function App() {
  const [activePage, setActivePage] = useState<PageName>("Overview");
  const [isDark, setIsDark] = useState(true);
  const [connectionState, setConnectionState] = useState<ConnectionState>("checking");
  const [apiError, setApiError] = useState("");

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark);
  }, [isDark]);

  useEffect(() => {
    const controller = new AbortController();
    setConnectionState("checking");
    getHealth(controller.signal)
      .then(() => {
        setApiError("");
        setConnectionState("connected");
      })
      .catch((error: unknown) => {
        if (error instanceof Error && error.name === "AbortError") return;
        setApiError(error instanceof ApiError || error instanceof Error ? error.message : "Unknown API error.");
        setConnectionState("error");
      });

    return () => controller.abort();
  }, []);

  const activeItem = navigation.find((item) => item.label === activePage) ?? navigation[0];

  return (
    <div className="min-h-screen bg-canvas text-ink transition-colors duration-200 dark:bg-[#101511] dark:text-[#edf1e8]">
      <div className="pointer-events-none fixed inset-0 overflow-hidden" aria-hidden="true">
        <div className="ambient-glow absolute -right-36 -top-40 h-[28rem] w-[28rem] rounded-full bg-lime-300/10 blur-3xl dark:bg-lime-300/[0.07]" />
      </div>
      <div className="relative mx-auto flex min-h-screen max-w-[1600px]">
        <aside className="hidden w-[248px] shrink-0 flex-col border-r border-black/[0.07] px-5 py-7 dark:border-white/[0.08] lg:flex">
          <Brand />
          <div className="mt-12 px-3 text-[10px] font-semibold uppercase tracking-[0.22em] text-ink/40 dark:text-white/35">Workspace</div>
          <nav className="mt-3 space-y-1" aria-label="Main navigation">
            {navigation.map((item) => <NavItem key={item.label} item={item} active={activePage === item.label} onClick={() => setActivePage(item.label)} />)}
          </nav>
          <div className="mt-auto">
            <div className="rounded-2xl border border-black/[0.07] bg-white/55 p-4 dark:border-white/[0.08] dark:bg-white/[0.035]">
              <div className="flex items-center gap-2 text-xs font-medium"><CircleHelp size={15} className="text-ink/50 dark:text-white/45" />Built for clarity</div>
              <p className="mt-2 text-xs leading-5 text-ink/55 dark:text-white/45">Every chart is tied to the analysis pipeline, not placeholder numbers.</p>
            </div>
            <div className="mt-5 flex items-center gap-2 px-2 text-[11px] text-ink/40 dark:text-white/35"><Command size={13} /><span>WattCast research workspace</span></div>
          </div>
        </aside>

        <main className="min-w-0 flex-1">
          <header className="sticky top-0 z-20 border-b border-black/[0.07] bg-canvas/85 px-5 py-4 backdrop-blur-xl dark:border-white/[0.08] dark:bg-[#101511]/85 sm:px-8 lg:px-10">
            <div className="flex items-center justify-between gap-4">
              <div className="flex min-w-0 items-center gap-3 lg:hidden"><Brand compact /></div>
              <div className="hidden min-w-0 lg:block"><p className="text-xs text-ink/45 dark:text-white/40">Energy intelligence / 01</p><p className="mt-1 text-sm font-medium">{activePage}</p></div>
              <div className="flex items-center gap-3">
                <ConnectionPill state={connectionState} />
                <button type="button" onClick={() => setIsDark((value) => !value)} className="icon-button" aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"} title={isDark ? "Switch to light mode" : "Switch to dark mode"}>
                  {isDark ? <Sun size={17} /> : <Moon size={17} />}
                </button>
              </div>
            </div>
            <nav className="mt-4 flex gap-1 overflow-x-auto pb-0.5 lg:hidden" aria-label="Mobile navigation">
              {navigation.map((item) => <button key={item.label} type="button" onClick={() => setActivePage(item.label)} className={`mobile-nav-item ${activePage === item.label ? "mobile-nav-item-active" : ""}`}><item.icon size={14} /><span>{item.label}</span></button>)}
            </nav>
          </header>

          <div className="px-5 pb-12 pt-8 sm:px-8 sm:pt-10 lg:px-10 lg:pt-12">
            <AnimatePresence mode="wait">
              <motion.section key={activePage} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.18, ease: "easeOut" }}>
                <div className="mb-8 flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
                  <div>
                    <div className="mb-3 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-ink/45 dark:text-white/40"><span className="h-1.5 w-1.5 rounded-full bg-lime-500" />Time-series analysis</div>
                    <h1 className="max-w-3xl text-3xl font-semibold tracking-[-0.045em] sm:text-4xl lg:text-[44px]">{activePage === "Overview" ? <>Understand the rhythm<br className="hidden sm:block" /> of energy use.</> : activePage}</h1>
                    <p className="mt-3 max-w-xl text-sm leading-6 text-ink/55 dark:text-white/50">{activePage === "Overview" ? "A grounded workspace for exploring household electricity patterns and evaluating forecasts." : activeItem.description}</p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2 self-start rounded-full border border-black/[0.08] bg-white/55 px-3 py-2 text-xs text-ink/60 dark:border-white/[0.09] dark:bg-white/[0.035] dark:text-white/55 sm:self-auto"><Activity size={14} /><span>Pipeline-backed analysis</span></div>
                </div>

                {connectionState === "error" && <div className="mb-6 rounded-2xl border border-rose-500/25 bg-rose-500/[0.06] p-5 sm:p-6" role="alert"><h2 className="font-medium">The API is not reachable</h2><p className="mt-2 break-words text-sm leading-6 text-ink/60 dark:text-white/55">{apiError}</p><p className="mt-2 text-xs text-ink/45 dark:text-white/40">Start FastAPI from the repository root, or update <code className="font-mono">VITE_API_BASE_URL</code> in <code className="font-mono">frontend/.env</code>.</p></div>}

                {activePage === "Overview" ? <OverviewPage /> : activePage === "Data explorer" ? <DataExplorerPage /> : activePage === "Correlation analysis" ? <CorrelationPage /> : <ComingSoonPage page={activePage} />}

                <div className="mt-8 flex flex-col gap-2 border-t border-black/[0.07] pt-5 text-xs text-ink/40 dark:border-white/[0.08] dark:text-white/35 sm:flex-row sm:items-center sm:justify-between">
                  <span>WattCast · Household energy time-series research</span>
                  <span className="inline-flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-lime-500" />Numbers appear only when sourced from the API</span>
                </div>
              </motion.section>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  );
}

function Brand({ compact = false }: { compact?: boolean }) {
  return <div className="flex items-center gap-3"><div className="flex h-10 w-10 items-center justify-center rounded-[14px] bg-accent text-[#24331a] shadow-sm"><Zap size={21} fill="currentColor" strokeWidth={2.2} /></div><div><div className="text-[17px] font-semibold tracking-[-0.045em]">wattcast<span className="text-lime-600 dark:text-lime-300">.</span></div>{!compact && <div className="mt-0.5 text-[10px] tracking-[0.12em] text-ink/40 dark:text-white/35">ENERGY INTELLIGENCE</div>}</div></div>;
}

function NavItem({ item, active, onClick }: { item: (typeof navigation)[number]; active: boolean; onClick: () => void }) {
  const Icon = item.icon;
  return <button type="button" onClick={onClick} aria-current={active ? "page" : undefined} className={`group flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-[13px] transition-colors ${active ? "bg-[#202a20] text-white dark:bg-white/[0.09] dark:text-white" : "text-ink/55 hover:bg-black/[0.035] hover:text-ink dark:text-white/50 dark:hover:bg-white/[0.045] dark:hover:text-white"}`}><Icon size={17} strokeWidth={active ? 2.1 : 1.7} className={active ? "text-lime-300" : "opacity-75"} /><span className="flex-1">{item.label}</span>{active && <ChevronRight size={14} className="text-lime-300" />}</button>;
}

function ConnectionPill({ state }: { state: ConnectionState }) {
  const label = state === "connected" ? "API connected" : state === "error" ? "API offline" : "Connecting";
  const dot = state === "connected" ? "bg-emerald-500" : state === "error" ? "bg-rose-500" : "animate-pulse bg-amber-400";
  return <div className="inline-flex items-center gap-2 rounded-full border border-black/[0.08] bg-white/50 px-3 py-2 text-[11px] font-medium text-ink/60 dark:border-white/[0.09] dark:bg-white/[0.035] dark:text-white/60"><span className={`h-1.5 w-1.5 rounded-full ${dot}`} /><span className="hidden sm:inline">{label}</span><span className="sm:hidden">{state === "connected" ? "Online" : state === "error" ? "Offline" : "Wait…"}</span></div>;
}

function ComingSoonPage({ page }: { page: string }) {
  return <section className="surface-card"><p className="eyebrow">Planned next</p><h2 className="mt-2 text-xl font-semibold">{page} is not implemented yet</h2><p className="mt-3 max-w-xl text-sm leading-6 text-ink/55 dark:text-white/50">This navigation entry is reserved for a later project step. The current build implements the Overview and Data explorer views using real API responses.</p></section>;
}
