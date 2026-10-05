import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useOverview, useSystemInfo } from "../api/hooks";
import { fmtDateTime } from "../lib/format";
import { useLive } from "../lib/stream";

type NavItem = { to: string; label: string; icon: string; end?: boolean; match?: string };
const ICON = {
  grid: "M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z",
  well: "M12 2v20M8 6h8M9 22h6M12 6c-3 3-3 9 0 12M12 6c3 3 3 9 0 12",
  bars: "M4 20V10M10 20V4M16 20v-8M22 20H2",
  wave: "M3 12c3-8 6-8 9 0s6 8 9 0",
  target: "M12 3v3M12 18v3M3 12h3M18 12h3M12 8a4 4 0 100 8 4 4 0 000-8z",
  db: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6",
  history: "M3 12a9 9 0 103-6.7M3 4v5h5M12 7v5l3 3",
  gauge: "M4 18a8 8 0 1116 0M12 18l4-6",
  bell: "M6 16V11a6 6 0 1112 0v5l2 2H4zM10 21h4",
  pulse: "M3 12h4l2-6 4 12 2-6h6",
};
const NAV_GROUPS: { title: string; note: string; items: NavItem[] }[] = [
  {
    title: "Field history",
    note: "300 wells · dataset v1.4",
    items: [
      { to: "/", label: "Field Overview", icon: ICON.grid, end: true },
      { to: "/history/WELL-001", label: "Well History", icon: ICON.history, match: "/history" },
      { to: "/css/WELL-001", label: "CSS Cycle Optimizer", icon: ICON.bars, match: "/css" },
      { to: "/spm/WELL-010", label: "SPM Advisor", icon: ICON.gauge, match: "/spm" },
      { to: "/warning", label: "Failure Early Warning", icon: ICON.bell },
      { to: "/dataset", label: "Dataset & Validation", icon: ICON.db },
    ],
  },
  {
    title: "Live twin",
    note: "12 wells · real-time simulation",
    items: [
      { to: "/live", label: "Live Field", icon: ICON.pulse },
      { to: "/wells/BGW-01", label: "Well Digital Twin", icon: ICON.well, match: "/wells" },
      { to: "/planner", label: "CSS Planner", icon: ICON.bars },
      { to: "/diagnostics", label: "SRP Diagnostics", icon: ICON.wave },
      { to: "/optimizer", label: "Optimizer & XAI", icon: ICON.target },
      { to: "/data", label: "Ingest & Models", icon: ICON.db },
    ],
  },
];
const NAV: NavItem[] = NAV_GROUPS.flatMap((g) => g.items);
const LIVE_PREFIXES = ["/live", "/wells", "/planner", "/diagnostics", "/optimizer", "/data"];

function Icon({ d }: { d: string }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={d} />
    </svg>
  );
}

export function Layout() {
  const { connected, simTs } = useLive();
  const { data: ov } = useOverview();
  const { data: info } = useSystemInfo();
  const mode = ov?.control_mode ?? info?.control_mode;
  const alerts = ov?.kpis.active_alerts ?? 0;
  const red = ov?.kpis.status_counts.red ?? 0;
  const { pathname } = useLocation();
  const live = LIVE_PREFIXES.some((p) => pathname === p || pathname.startsWith(`${p}/`));
  const isActive = (n: NavItem, active: boolean) => active || (!!n.match && pathname.startsWith(n.match));

  return (
    <div className="flex h-full">
      <aside className="hidden w-[232px] shrink-0 flex-col border-r border-line bg-panel md:flex">
        <div className="flex items-center gap-2.5 px-4 py-4">
          <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden>
            <rect width="32" height="32" rx="7" fill="#0e1318" stroke="#263140" />
            <path d="M3 17h6l3-9 5 17 3-11 2 3h7" fill="none" stroke="#4cc3d9" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div className="leading-tight">
            <div className="text-[15px] font-bold tracking-[0.18em] text-ink">PULSE</div>
            <div className="text-[10px] uppercase tracking-wider text-muted">Lift &amp; Steam Engine</div>
          </div>
        </div>
        <nav className="flex-1 space-y-4 overflow-y-auto px-2 py-2" aria-label="Main">
          {NAV_GROUPS.map((g) => (
            <div key={g.title}>
              <div className="px-3 pb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-faint">
                {g.title}
                <span className="block font-normal normal-case tracking-normal">{g.note}</span>
              </div>
              <div className="space-y-0.5">
                {g.items.map((n) => (
                  <NavLink
                    key={n.to}
                    to={n.to}
                    end={n.end}
                    className={({ isActive: a }) =>
                      `flex items-center gap-3 rounded-md px-3 py-1.5 text-[13px] font-medium transition ${isActive(n, a) ? "bg-accent/12 text-accent" : "text-muted hover:bg-panel2 hover:text-ink"}`
                    }
                  >
                    <Icon d={n.icon} />
                    {n.label}
                  </NavLink>
                ))}
              </div>
            </div>
          ))}
        </nav>
        <div className="border-t border-line px-4 py-3 text-[11px] leading-relaxed text-faint">
          Baghewala Field · CSS
          <br />
          SIH26120 · Team Fluxora (178187)
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-4 border-b border-line bg-panel px-4">
          <nav className="flex gap-1 overflow-x-auto md:hidden" aria-label="Main (compact)">
            {NAV.map((n) => (
              <NavLink key={n.to} to={n.to} end={n.end} title={n.label} className={({ isActive: a }) => `rounded-md p-2 ${isActive(n, a) ? "bg-accent/12 text-accent" : "text-muted"}`}>
                <Icon d={n.icon} />
              </NavLink>
            ))}
          </nav>
          <div className="hidden text-[13px] text-muted md:block">
            Baghewala Field <span className="mx-1.5 text-faint">/</span> <span className="text-ink">{live ? "Live twin (simulated)" : "Field history (dataset)"}</span>
          </div>
          <div className="ml-auto flex items-center gap-4 text-xs">
            {live && red > 0 && (
              <span className="flex items-center gap-1.5 rounded-full border border-bad/50 bg-bad/10 px-2.5 py-1 font-semibold text-bad" role="status">
                <span className="blink-red inline-block h-2 w-2 rounded-full bg-bad" /> {red} critical · {alerts} active alert{alerts === 1 ? "" : "s"}
              </span>
            )}
            {mode && (
              <span
                title={mode === "advisory" ? "Advisory mode: every set-point needs operator approval" : "Closed loop: safe advisories are applied automatically"}
                className={`rounded-md border px-2 py-1 font-semibold uppercase tracking-wide ${mode === "advisory" ? "border-accent/40 text-accent" : "border-warn/50 text-warn"}`}
              >
                {mode === "advisory" ? "Advisory mode" : "Closed loop"}
              </span>
            )}
            {live && (
              <span className="num hidden text-muted sm:inline" title="Field clock (simulated historian time)">
                {fmtDateTime(simTs ?? ov?.kpis.sim_ts)}
              </span>
            )}
            <span className="flex items-center gap-1.5" role="status" aria-live="polite">
              <span className={`inline-block h-2 w-2 rounded-full ${connected ? "bg-ok" : "bg-bad"}`} />
              <span className={connected ? "text-muted" : "text-bad"}>{connected ? "API live" : "Reconnecting…"}</span>
            </span>
          </div>
        </header>
        {live ? (
          info?.data_provenance?.startsWith("synthetic") && (
            <div className="border-b border-warn/25 bg-warn/8 px-4 py-1 text-[11px] text-warn/90">
              Live twin: 12 simulated wells with physics-generated telemetry and dynamometer cards (CNN trained on simulated cards). Connect a SCADA/historian feed for field use.
            </div>
          )
        ) : (
          <div className="border-b border-accent/20 bg-accent/6 px-4 py-1 text-[11px] text-accent/90">
            Field history: PULSE synthetic dataset v1.4 (300 wells, physics generator calibrated to published Baghewala figures; no Oil India well data). Models are scored on held-out wells.
          </div>
        )}
        <main className="min-h-0 flex-1 overflow-y-auto p-4 md:p-5">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
