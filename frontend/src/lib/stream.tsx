import { useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { wsUrl } from "../api/client";
import type { DynoCard, Kpis, Status } from "../api/types";

export interface LiveTickWell {
  id: string;
  status: Status;
  phase: string;
  spm: number;
  oil_rate_m3d: number;
  visc_cp: number;
  kw: number;
  producing: boolean;
  diag: string | null;
}
export interface LiveEvent {
  key: string;
  ts: number;
  kind: "alert" | "alert_cleared" | "advisory" | "job" | "ingest";
  text: string;
  severity: "red" | "amber" | "info";
  wellId?: string;
}

interface Live {
  connected: boolean;
  simTs: number | null;
  kpis: Kpis | null;
  wells: Record<string, LiveTickWell>;
  cards: Record<string, DynoCard>;
  lastCard: { card: DynoCard; at: number } | null;
  events: LiveEvent[];
  subscribe: (wells: string[]) => void;
}

const Ctx = createContext<Live | null>(null);
const MAX_EVENTS = 40;

export function LiveProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [connected, setConnected] = useState(false);
  const [simTs, setSimTs] = useState<number | null>(null);
  const [kpis, setKpis] = useState<Kpis | null>(null);
  const [wells, setWells] = useState<Record<string, LiveTickWell>>({});
  const [cards, setCards] = useState<Record<string, DynoCard>>({});
  const [lastCard, setLastCard] = useState<{ card: DynoCard; at: number } | null>(null);
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const subsRef = useRef<string[]>([]);
  const lastInvalidate = useRef(0);

  const send = useCallback((obj: unknown) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }, []);

  const subscribe = useCallback(
    (list: string[]) => {
      const same = list.length === subsRef.current.length && list.every((w, i) => w === subsRef.current[i]);
      if (same) return;
      subsRef.current = list;
      send({ action: "subscribe", wells: list });
    },
    [send],
  );

  useEffect(() => {
    let stopped = false;
    let retry = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let ping: ReturnType<typeof setInterval> | undefined;

    const pushEvent = (e: Omit<LiveEvent, "key">) =>
      setEvents((prev) => [{ ...e, key: `${e.ts}-${Math.random().toString(36).slice(2, 7)}` }, ...prev].slice(0, MAX_EVENTS));

    const connect = () => {
      if (stopped) return;
      const ws = new WebSocket(wsUrl());
      wsRef.current = ws;
      ws.onopen = () => {
        retry = 0;
        setConnected(true);
        if (subsRef.current.length) ws.send(JSON.stringify({ action: "subscribe", wells: subsRef.current }));
        ping = setInterval(() => ws.readyState === WebSocket.OPEN && ws.send('{"action":"ping"}'), 25000);
      };
      ws.onmessage = (ev) => {
        let m: Record<string, unknown>;
        try {
          m = JSON.parse(ev.data as string);
        } catch {
          return;
        }
        switch (m.type) {
          case "hello":
            setSimTs(m.sim_ts as number);
            break;
          case "tick": {
            setSimTs(m.ts as number);
            setKpis(m.kpis as Kpis);
            const next: Record<string, LiveTickWell> = {};
            for (const w of m.wells as LiveTickWell[]) next[w.id] = w;
            setWells(next);
            const now = Date.now();
            if (now - lastInvalidate.current > 2500) {
              lastInvalidate.current = now;
              qc.invalidateQueries({ queryKey: ["overview"] });
              qc.invalidateQueries({ queryKey: ["telemetry"] });
              qc.invalidateQueries({ queryKey: ["well-detail"] });
            }
            break;
          }
          case "dyno": {
            const c = m as unknown as DynoCard;
            setCards((prev) => ({ ...prev, [c.well_id]: c }));
            setLastCard({ card: c, at: Date.now() });
            break;
          }
          case "alert":
            pushEvent({ ts: m.ts as number, kind: "alert", text: m.message as string, severity: m.severity as "red" | "amber", wellId: m.well_id as string });
            qc.invalidateQueries({ queryKey: ["alerts"] });
            break;
          case "alert_cleared":
            pushEvent({ ts: m.ts as number, kind: "alert_cleared", text: `Alert cleared on ${m.well_id}`, severity: "info", wellId: m.well_id as string });
            qc.invalidateQueries({ queryKey: ["alerts"] });
            break;
          case "advisory": {
            const a = m.advisory as { status: string; recommended_spm: number; current_spm: number } | null;
            if (a) {
              pushEvent({
                ts: (m.advisory as { ts: number }).ts,
                kind: "advisory",
                text: `Advisory ${a.status} for ${m.well_id}: ${a.current_spm.toFixed(1)} → ${a.recommended_spm.toFixed(1)} SPM`,
                severity: "info",
                wellId: m.well_id as string,
              });
            }
            qc.invalidateQueries({ queryKey: ["advisories"] });
            qc.invalidateQueries({ queryKey: ["recommendation"] });
            qc.invalidateQueries({ queryKey: ["audit"] });
            break;
          }
          case "job":
            qc.invalidateQueries({ queryKey: ["job", m.id] });
            break;
          case "ingest":
            pushEvent({ ts: Math.floor(Date.now() / 1000), kind: "ingest", text: `Ingested ${m.rows} ${m.kind} rows`, severity: "info" });
            qc.invalidateQueries({ queryKey: ["telemetry"] });
            break;
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (ping) clearInterval(ping);
        if (!stopped) {
          retry += 1;
          timer = setTimeout(connect, Math.min(1000 * 2 ** Math.min(retry, 4), 15000));
        }
      };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
      if (ping) clearInterval(ping);
      wsRef.current?.close();
    };
  }, [qc]);

  const value = useMemo(() => ({ connected, simTs, kpis, wells, cards, lastCard, events, subscribe }), [connected, simTs, kpis, wells, cards, lastCard, events, subscribe]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useLive(): Live {
  const v = useContext(Ctx);
  if (!v) throw new Error("useLive must be used inside <LiveProvider>");
  return v;
}

/** Subscribe the socket to dyno cards of the given wells while the component is mounted. */
export function useCardSubscription(wells: string[]) {
  const { subscribe } = useLive();
  const key = wells.join(",");
  useEffect(() => {
    subscribe(wells);
    return () => subscribe([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, subscribe]);
}
