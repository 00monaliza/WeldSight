import type { Inspector } from "../hooks/useInspector";

export type Tab = "inspect" | "results" | "api";
const LABELS: Record<Tab, string> = { inspect: "Inspect", results: "Stage 1 results", api: "API" };

export function Header({ tab, insp }: { tab: Tab; insp: Inspector }) {
  const { settings, health } = insp;
  const api = settings.engine === "api";
  const chip = !api ? "Demo engine" : health?.ok ? (health.info.stub ? "API · stub" : `API · ${health.info.model}`) : health ? "API · offline" : "API · …";
  const mode = api && health?.ok && !health.info.stub ? "live" : api && health && !health.ok ? "down" : "demo";
  return (
    <header className="top">
      <a className="brand" href="#inspect" aria-label="WeldSight">
        <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true">
          <circle cx="16" cy="16" r="14" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <path d="M2 16h8l3-7 4 14 3-7h10" fill="none" stroke="var(--amber)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        </svg>
        <span>WELD<b>SIGHT</b></span>
      </a>
      <nav className="tabs" role="tablist">
        {(Object.keys(LABELS) as Tab[]).map((t) => (
          <a key={t} href={`#${t}`} role="tab" aria-selected={tab === t}>{LABELS[t]}</a>
        ))}
      </nav>
      <div className="engine-chip" data-mode={mode} title={health?.ok ? health.info.model : undefined}><i /><span>{chip}</span></div>
    </header>
  );
}
