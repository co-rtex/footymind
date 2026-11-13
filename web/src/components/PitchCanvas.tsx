import { useEffect, useRef, type PointerEvent } from "react";
import type { ReplayData } from "../api/parse";
import { INTERVENTION_BOUNDS } from "../api/types";
import { clamp, pitchAspect } from "../lib/coords";
import { PitchRenderer, type DragState, type DrawnPlayer } from "../render/renderer";
import { readTheme } from "../render/theme";

export interface DragInit {
  fromX: number;
  fromY: number;
  originX: number | null;
  originY: number | null;
  radius: number | null;
}

interface Props {
  replay: ReplayData;
  editable: boolean;
  ariaLabel: string;
  describedBy?: string;
  onReady: (r: PitchRenderer | null) => void;
  onPick: (id: string | null) => void;
  /** Return reach info to allow dragging this player, or null to refuse. */
  onDragStart: (p: DrawnPlayer) => DragInit | null;
  onDragOutsideChange: (outside: boolean | null, playerId: string | null) => void;
  onDrop: (playerId: string, x: number, y: number) => void;
}

/** Players may be dropped up to 2 m outside the lines, as in the backend feasibility check. */
const DROP_MARGIN_M = 2;

export function PitchCanvas({
  replay,
  editable,
  ariaLabel,
  describedBy,
  onReady,
  onPick,
  onDragStart,
  onDragOutsideChange,
  onDrop,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<PitchRenderer | null>(null);
  const drag = useRef<{
    id: string;
    sx: number;
    sy: number;
    moved: boolean;
    init: DragInit;
    outside: boolean | null;
    last: DragState | null;
  } | null>(null);
  const press = useRef<{ id: string | null; sx: number; sy: number } | null>(null);

  // the latest callbacks without re-binding listeners
  const cb = useRef({ onPick, onDragStart, onDragOutsideChange, onDrop, editable });
  cb.current = { onPick, onDragStart, onDragOutsideChange, onDrop, editable };

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const r = new PitchRenderer(canvas, readTheme());
    r.setScene({ replay }, false);
    rendererRef.current = r;
    const aspect = pitchAspect(replay.pitch);
    const fit = () => {
      const w = Math.max(240, Math.floor(wrap.clientWidth));
      r.resize(w, Math.round(w * aspect));
    };
    fit();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(fit) : null;
    ro?.observe(wrap);
    // re-rasterise when the device pixel ratio changes (zoom, moving between screens)
    const mq = typeof matchMedia !== "undefined" ? matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`) : null;
    mq?.addEventListener?.("change", fit);
    onReady(r);
    return () => {
      ro?.disconnect();
      mq?.removeEventListener?.("change", fit);
      r.destroy();
      rendererRef.current = null;
      onReady(null);
    };
    // the renderer lives for the lifetime of this replay
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replay]);

  const local = (e: PointerEvent<HTMLCanvasElement>): [number, number] => {
    const rect = e.currentTarget.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  };

  const onPointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    const r = rendererRef.current;
    if (!r || e.button !== 0) return;
    const [px, py] = local(e);
    const hit = r.pick(px, py);
    press.current = { id: hit?.id ?? null, sx: px, sy: py };
    if (cb.current.editable && hit) {
      const init = cb.current.onDragStart(hit);
      if (init) {
        drag.current = { id: hit.id, sx: px, sy: py, moved: false, init, outside: null, last: null };
        e.currentTarget.setPointerCapture(e.pointerId);
        e.preventDefault();
      }
    }
  };

  const onPointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const r = rendererRef.current;
    if (!r) return;
    const [px, py] = local(e);
    const d = drag.current;
    if (!d) {
      const over = r.pick(px, py);
      e.currentTarget.style.cursor = over ? (cb.current.editable ? "grab" : "pointer") : "default";
      return;
    }
    if (!d.moved && Math.hypot(px - d.sx, py - d.sy) < 3) return;
    d.moved = true;
    e.currentTarget.style.cursor = "grabbing";
    const p = r.toPitch(px, py);
    if (!p) return;
    const hl = replay.pitch.length / 2 + DROP_MARGIN_M;
    const hw = replay.pitch.width / 2 + DROP_MARGIN_M;
    const x = clamp(p[0], -Math.min(hl, INTERVENTION_BOUNDS.x), Math.min(hl, INTERVENTION_BOUNDS.x));
    const y = clamp(p[1], -Math.min(hw, INTERVENTION_BOUNDS.y), Math.min(hw, INTERVENTION_BOUNDS.y));
    const { originX, originY, radius } = d.init;
    const outside =
      originX !== null && originY !== null && radius !== null ? Math.hypot(x - originX, y - originY) > radius : null;
    const state: DragState = { playerId: d.id, x, y, fromX: d.init.fromX, fromY: d.init.fromY, originX, originY, radius, outside };
    d.last = state;
    r.setScene({ drag: state });
    if (outside !== d.outside) {
      d.outside = outside;
      cb.current.onDragOutsideChange(outside, d.id);
    }
  };

  const finish = (e: PointerEvent<HTMLCanvasElement>, cancelled: boolean) => {
    const r = rendererRef.current;
    const d = drag.current;
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    e.currentTarget.style.cursor = "default";
    if (d) {
      r?.setScene({ drag: null });
      cb.current.onDragOutsideChange(null, null);
      if (!cancelled && d.moved && d.last) {
        cb.current.onDrop(d.id, d.last.x, d.last.y);
        press.current = null;
        return;
      }
    }
    const pr = press.current;
    press.current = null;
    if (!cancelled && pr) {
      const [px, py] = local(e);
      if (Math.hypot(px - pr.sx, py - pr.sy) < 5) cb.current.onPick(pr.id);
    }
  };

  return (
    <div className="pitch-wrap" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        className={editable ? "pitch-canvas is-editable" : "pitch-canvas"}
        role="img"
        aria-label={ariaLabel}
        aria-describedby={describedBy}
        data-testid="pitch-canvas"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={(e) => finish(e, false)}
        onPointerCancel={(e) => finish(e, true)}
      />
    </div>
  );
}
