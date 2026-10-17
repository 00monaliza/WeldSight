import { useEffect, useState } from "react";
import { ApiView } from "./components/ApiView";
import { Header, type Tab } from "./components/Header";
import { InspectView } from "./components/InspectView";
import { ResultsView } from "./components/ResultsView";
import { useInspector } from "./hooks/useInspector";

const TABS: Tab[] = ["inspect", "results", "api"];
const fromHash = (): Tab => {
  const t = location.hash.slice(1) as Tab;
  return TABS.includes(t) ? t : "inspect";
};

export default function App() {
  const [tab, setTab] = useState<Tab>(fromHash);
  const insp = useInspector();
  useEffect(() => {
    const on = () => setTab(fromHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  // views stay mounted so the film and results survive tab switches
  return (
    <>
      <Header tab={tab} insp={insp} />
      <main>
        <section className="view" hidden={tab !== "inspect"}><InspectView insp={insp} /></section>
        <section className="view" hidden={tab !== "results"}><ResultsView /></section>
        <section className="view" hidden={tab !== "api"}><ApiView /></section>
      </main>
      <footer className="foot">WeldSight · Stage 1 research interface · weld-defect segmentation with uncertainty on real radiographs</footer>
    </>
  );
}
