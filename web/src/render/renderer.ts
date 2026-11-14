/**
 * Imperative Canvas 2D renderer for the tactical pitch.
 *
 * React never re-renders per animation frame: the playback loop writes `t`
 * into the scene and calls drawNow(). Everything else (analysis, scenario,
 * toggles) arrives through setScene(), which coalesces into one rAF draw.
 */
import type {
  DefensiveShadowOut,
  FrameAnalysisOut,
  PassOptionOut,
  ProvenanceLabel,
  ScenarioResponse,
  Team,
} from "../api/types";
import type { ReplayData } from "../api/parse";
import { fitPitch, metresToPx, toCanvasX, toCanvasY, toPitch, type PitchTransform } from "../lib/coords";
import { ESTIMATED, NOT_DRAWN, sampleTrack, sampleXY, segmentEstimated, type Vec2 } from "../lib/interp";
import { primaryP } from "../lib/models";
import { shadowHalo, suppressedReceiver } from "../lib/shadow";
import { drawPitch } from "./pitch";
import { discStyle, ESTIMATED_TRAIL, styleKindFor } from "./styles";
import { controlDelta, controlRaster, deltaRaster } from "./heatmap";
import { DEFAULT_THEME, rgba, type CanvasTheme } from "./theme";

export type ScenarioView = "observed" | "hypothetical" | "difference";

export interface DragState {
  playerId: string;
  x: number;
  y: number;
  fromX: number;
  fromY: number;
  originX: number | null;
  originY: number | null;
  radius: number | null;
  /** true when the drop point lies outside the reachable circle, null if unknown */
  outside: boolean | null;
}

export interface Layers {
  control: boolean;
  passes: boolean;
  pressure: boolean;
  trails: boolean;
  /** experimental defensive shadow; off by default */
  shadow: boolean;
}

export interface Scene {
  replay: ReplayData | null;
  /** fractional replay frame index */
  t: number;
  playing: boolean;
  analysis: FrameAnalysisOut | null;
  scenario: ScenarioResponse | null;
  view: ScenarioView;
  layers: Layers;
  editMode: boolean;
  drag: DragState | null;
  /** user moves not yet reflected in `scenario` (request in flight) */
  pending: ReadonlyMap<string, { x: number; y: number }>;
  selectedId: string | null;
  highlightReceiver: string | null;
  /** players named by the focused "why this moment" statement, ringed on the pitch */
  tracePlayers: readonly string[];
  /** team display names for the attack-direction labels */
  teams: { home: string; away: string };
  /** defensive shadow for the paused frame (experimental) */
  shadow: DefensiveShadowOut | null;
}

export interface DrawnPlayer {
  id: string;
  team: Team;
  jersey: number;
  x: number;
  y: number;
  prov: ProvenanceLabel;
}

const EMPTY_PENDING: ReadonlyMap<string, { x: number; y: number }> = new Map();

export function defaultScene(): Scene {
  return {
    replay: null,
    t: 0,
    playing: false,
    analysis: null,
    scenario: null,
    view: "observed",
    layers: { control: true, passes: true, pressure: true, trails: true, shadow: false },
    editMode: false,
    drag: null,
    pending: EMPTY_PENDING,
    selectedId: null,
    highlightReceiver: null,
    tracePlayers: [],
    teams: { home: "Home", away: "Away" },
    shadow: null,
  };
}

const TRAIL_SECONDS = 1.5;

export class PitchRenderer {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D | null;
  private theme: CanvasTheme;
  private scene: Scene = defaultScene();
  private tf: PitchTransform | null = null;
  private dpr = 1;
  private bg: HTMLCanvasElement | null = null;
  private raf = 0;
  private heat: { key: unknown; view: ScenarioView; team: Team | null; canvas: HTMLCanvasElement | null } | null = null;
  private drawn: DrawnPlayer[] = [];
  private ball: Vec2 | null = null;
  private tmp: Vec2 = { x: 0, y: 0 };

  constructor(canvas: HTMLCanvasElement, theme: CanvasTheme = DEFAULT_THEME) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.theme = theme;
  }

  setTheme(theme: CanvasTheme): void {
    this.theme = theme;
    this.bg = null;
    this.heat = null;
    this.invalidate();
  }

  getScene(): Readonly<Scene> {
    return this.scene;
  }

  /** Merge scene fields and schedule a redraw (coalesced to one per frame). */
  setScene(patch: Partial<Scene>, redraw = true): void {
    if (patch.replay !== undefined && patch.replay !== this.scene.replay) {
      this.bg = null;
      if (patch.replay && this.tf) this.tf = fitPitch(this.tf.cssWidth, this.tf.cssHeight, patch.replay.pitch);
    }
    this.scene = { ...this.scene, ...patch };
    if (redraw) this.invalidate();
  }

  resize(cssWidth: number, cssHeight: number, dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1): void {
    const pitch = this.scene.replay?.pitch ?? { length: 105, width: 68 };
    this.dpr = Math.min(3, Math.max(1, dpr));
    this.canvas.width = Math.round(cssWidth * this.dpr);
    this.canvas.height = Math.round(cssHeight * this.dpr);
    this.canvas.style.width = `${cssWidth}px`;
    this.canvas.style.height = `${cssHeight}px`;
    this.tf = fitPitch(cssWidth, cssHeight, pitch);
    this.bg = null;
    this.drawNow();
  }

  invalidate(): void {
    if (this.raf || typeof requestAnimationFrame === "undefined") return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.drawNow();
    });
  }

  destroy(): void {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
  }

  transform(): PitchTransform | null {
    return this.tf;
  }

  toPitch(px: number, py: number): [number, number] | null {
    return this.tf ? toPitch(this.tf, px, py) : null;
  }

  /** Player under a CSS-pixel point, or null. */
  pick(px: number, py: number): DrawnPlayer | null {
    if (!this.tf) return null;
    const hit = Math.max(this.radiusPx() + 4, 12);
    let best: DrawnPlayer | null = null;
    let bestD = hit;
    for (const p of this.drawn) {
      const d = Math.hypot(toCanvasX(this.tf, p.x) - px, toCanvasY(this.tf, p.y) - py);
      if (d <= bestD) {
        bestD = d;
        best = p;
      }
    }
    return best;
  }

  drawnPlayers(): readonly DrawnPlayer[] {
    return this.drawn;
  }

  /** The analysis whose overlays apply right now, if any. */
  activeAnalysis(): FrameAnalysisOut | null {
    const s = this.scene;
    if (s.scenario) return s.view === "observed" ? s.scenario.observed : s.scenario.hypothetical;
    if (!s.analysis || s.playing || !s.replay) return null;
    const idx = Math.round(s.t);
    if (Math.abs(s.t - idx) > 1e-6 || s.replay.frame[idx] !== s.analysis.frame) return null;
    return s.analysis;
  }

  private radiusPx(): number {
    return this.tf ? Math.max(5, Math.min(14, 1.05 * this.tf.scale)) : 8;
  }

  private teamColour(team: Team): string {
    return team === "home" ? this.theme.home : this.theme.away;
  }

  drawNow(): void {
    const ctx = this.ctx;
    const tf = this.tf;
    if (!ctx || !tf) return;
    const s = this.scene;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.drawBackground(ctx, tf);
    this.collect();
    const active = this.activeAnalysis();
    const hypView = !!s.scenario && s.view !== "observed";
    const showScenarioLayer = s.editMode && s.view !== "observed";

    if (active && s.layers.control) this.drawHeat(ctx, tf, active);
    this.drawAttackDirections(ctx, tf);
    if (s.layers.trails && !hypView && !showScenarioLayer) this.drawTrails(ctx, tf);
    if (active && s.layers.passes && !s.drag) this.drawPassOptions(ctx, tf, active);
    if (active && s.layers.pressure && !s.drag) this.drawPressure(ctx, tf, active);
    const shadow = this.activeShadow();
    if (shadow) this.drawShadow(ctx, tf, shadow);
    if (showScenarioLayer || hypView) this.drawGhosts(ctx, tf);
    if (s.drag) this.drawReach(ctx, tf, s.drag);
    if (s.tracePlayers.length > 0) this.drawTrace(ctx, tf);
    this.drawPlayers(ctx, tf, active);
    this.drawBall(ctx, tf);
  }

  private drawBackground(ctx: CanvasRenderingContext2D, tf: PitchTransform): void {
    if (!this.bg && typeof document !== "undefined") {
      const bg = document.createElement("canvas");
      bg.width = this.canvas.width;
      bg.height = this.canvas.height;
      const bctx = bg.getContext("2d");
      if (bctx) {
        bctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
        drawPitch(bctx, tf, this.theme);
        this.bg = bg;
      }
    }
    if (this.bg) ctx.drawImage(this.bg, 0, 0, tf.cssWidth, tf.cssHeight);
    else drawPitch(ctx, tf, this.theme);
  }

  /** Build the list of players and the ball to draw for the current scene. */
  private collect(): void {
    const s = this.scene;
    const r = s.replay;
    this.drawn.length = 0;
    this.ball = null;
    if (!r) return;
    const hyp = s.scenario && s.view !== "observed" ? s.scenario.hypothetical : null;
    const overrideAllowed = s.editMode && s.view !== "observed";

    if (hyp) {
      for (const p of hyp.players) {
        if (p.x === null || p.y === null) continue;
        this.drawn.push({ id: p.player_id, team: p.team, jersey: p.jersey, x: p.x, y: p.y, prov: p.position_provenance });
      }
      if (hyp.ball_x !== null && hyp.ball_y !== null) this.ball = { x: hyp.ball_x, y: hyp.ball_y };
    } else {
      const v = this.tmp;
      for (let k = 0; k < r.nPlayers; k++) {
        const kind = sampleTrack(r.xy, r.nPlayers, r.nFrames, s.t, k, v, r.period, r.detected);
        if (kind === NOT_DRAWN) continue;
        const pl = r.players[k];
        if (!pl) continue;
        this.drawn.push({
          id: pl.player_id,
          team: pl.team,
          jersey: pl.jersey,
          x: v.x,
          y: v.y,
          prov: kind === ESTIMATED ? "estimated" : "observed",
        });
      }
      if (sampleXY(r.ball, 1, r.nFrames, s.t, 0, v, r.period)) this.ball = { x: v.x, y: v.y };
    }

    if (overrideAllowed) {
      for (const p of this.drawn) {
        const pend = s.pending.get(p.id);
        if (pend) {
          p.x = pend.x;
          p.y = pend.y;
          p.prov = "hypothetical";
        }
      }
    }
    if (s.drag) {
      const p = this.drawn.find((q) => q.id === s.drag?.playerId);
      if (p) {
        p.x = s.drag.x;
        p.y = s.drag.y;
        p.prov = "hypothetical";
      }
    }
  }

  private drawHeat(ctx: CanvasRenderingContext2D, tf: PitchTransform, active: FrameAnalysisOut): void {
    const s = this.scene;
    const att = active.attacking_team;
    if (!att) return;
    const def: Team = att === "home" ? "away" : "home";
    let key: unknown;
    let geomSrc = active.pitch_control;
    if (s.scenario && s.view === "difference") {
      key = s.scenario;
      geomSrc = s.scenario.hypothetical.pitch_control;
    } else {
      key = active.pitch_control;
    }
    if (!geomSrc) return;
    if (!this.heat || this.heat.key !== key || this.heat.view !== s.view || this.heat.team !== att) {
      let canvas: HTMLCanvasElement | null = null;
      if (s.scenario && s.view === "difference") {
        const o = s.scenario.observed.pitch_control;
        const h = s.scenario.hypothetical.pitch_control;
        const d = o && h ? controlDelta(o, h) : null;
        canvas = d && h ? deltaRaster(h, d, this.theme.positive, this.theme.negative) : null;
      } else {
        canvas = controlRaster(geomSrc, this.teamColour(att), this.teamColour(def), 0.34);
      }
      this.heat = { key, view: s.view, team: att, canvas };
    }
    const c = this.heat.canvas;
    if (!c) return;
    const g = geomSrc;
    const left = toCanvasX(tf, g.x0 - g.dx / 2);
    const top = toCanvasY(tf, g.y0 + (g.ny - 0.5) * g.dy);
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(c, left, top, g.nx * g.dx * tf.scale, g.ny * g.dy * tf.scale);
    ctx.restore();
  }

  /** The defensive shadow applies only to the paused, unedited frame it was computed for. */
  activeShadow(): DefensiveShadowOut | null {
    const s = this.scene;
    if (!s.layers.shadow || !s.shadow || s.playing || s.editMode || s.scenario || !s.replay) return null;
    const idx = Math.round(s.t);
    if (Math.abs(s.t - idx) > 1e-6 || s.replay.frame[idx] !== s.shadow.frame) return null;
    return s.shadow;
  }

  private drawShadow(ctx: CanvasRenderingContext2D, tf: PitchTransform, sh: DefensiveShadowOut): void {
    const colour = this.theme.shadow;
    ctx.save();
    for (const d of sh.defenders) {
      const halo = shadowHalo(d);
      if (!halo) continue; // below the noise floor or not suppressive: no ring
      const x = toCanvasX(tf, d.x);
      const y = toCanvasY(tf, d.y);
      const r = Math.max(this.radiusPx() + 6, metresToPx(tf, halo.radiusM));
      const g = ctx.createRadialGradient(x, y, this.radiusPx() * 0.6, x, y, r);
      g.addColorStop(0, rgba(colour, halo.alpha * 0.55));
      g.addColorStop(0.7, rgba(colour, halo.alpha * 0.22));
      g.addColorStop(1, rgba(colour, 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = rgba(colour, Math.min(1, halo.alpha + 0.15));
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.stroke();
      const rid = suppressedReceiver(d);
      const rec = this.drawnById(rid);
      if (rec) {
        const rx = toCanvasX(tf, rec.x);
        const ry = toCanvasY(tf, rec.y);
        const len = Math.hypot(rx - x, ry - y);
        if (len > this.radiusPx() * 2) {
          const ux = (rx - x) / len;
          const uy = (ry - y) / len;
          ctx.setLineDash([3, 4]);
          ctx.lineWidth = 1.25;
          ctx.strokeStyle = rgba(colour, 0.85);
          ctx.beginPath();
          ctx.moveTo(x + ux * (this.radiusPx() + 2), y + uy * (this.radiusPx() + 2));
          ctx.lineTo(rx - ux * (this.radiusPx() + 3), ry - uy * (this.radiusPx() + 3));
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }
    ctx.restore();
  }

  private drawAttackDirections(ctx: CanvasRenderingContext2D, tf: PitchTransform): void {
    const r = this.scene.replay;
    if (!r) return;
    const marginPx = (tf.cssHeight - r.pitch.width * tf.scale) / 2;
    if (marginPx < 14) return;
    const idx = Math.max(0, Math.min(r.nFrames - 1, Math.round(this.scene.t)));
    const signs = r.attackSign.get(r.period[idx] ?? 1);
    if (!signs) return;
    const y = toCanvasY(tf, -r.pitch.width / 2) + marginPx / 2;
    ctx.save();
    ctx.font = `600 ${Math.min(11, Math.max(9, marginPx * 0.36))}px ${fontStack()}`;
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    for (const team of ["home", "away"] as Team[]) {
      const sign = signs[team];
      if (!sign) continue;
      // label sits in the team's own half and points towards the goal it attacks
      const x = toCanvasX(tf, (-sign * r.pitch.length) / 4);
      const name = (team === "home" ? this.scene.teams.home : this.scene.teams.away).toUpperCase();
      const label = sign > 0 ? `${name} attacks →` : `← ${name} attacks`;
      ctx.fillStyle = rgba(this.teamColour(team), 0.85);
      ctx.fillText(label, x, y);
    }
    ctx.restore();
  }

  private drawTrails(ctx: CanvasRenderingContext2D, tf: PitchTransform): void {
    const s = this.scene;
    const r = s.replay;
    if (!r) return;
    const n = Math.max(2, Math.round((TRAIL_SECONDS * r.fps) / r.step));
    const end = Math.floor(s.t);
    const start = Math.max(0, end - n);
    if (end <= start) return;
    const buckets = 4;
    const lw = Math.max(1, Math.min(2.5, tf.scale * 0.22));
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = lw;
    for (const team of ["home", "away"] as Team[]) {
      const colour = this.teamColour(team);
      for (let b = 0; b < buckets; b++) {
        const i0 = start + Math.floor(((end - start) * b) / buckets);
        const i1 = start + Math.floor(((end - start) * (b + 1)) / buckets);
        const obs = new Path2D();
        const est = new Path2D();
        let anyEst = false;
        for (let k = 0; k < r.nPlayers; k++) {
          if (r.players[k]?.team !== team) continue;
          let prevX = NaN;
          let prevY = NaN;
          for (let i = i0; i <= i1; i++) {
            const o = (i * r.nPlayers + k) * 2;
            const x = r.xy[o] as number;
            const y = r.xy[o + 1] as number;
            const brk = i > 0 && r.period[i] !== r.period[i - 1];
            if (!Number.isFinite(x) || !Number.isFinite(y) || brk) {
              prevX = NaN;
              continue;
            }
            const px = toCanvasX(tf, x);
            const py = toCanvasY(tf, y);
            if (Number.isFinite(prevX)) {
              const path = segmentEstimated(r.detected, r.nPlayers, i, k) ? est : obs;
              if (path === est) anyEst = true;
              path.moveTo(prevX, prevY);
              path.lineTo(px, py);
            }
            prevX = px;
            prevY = py;
          }
        }
        const alpha = 0.08 + 0.13 * b;
        ctx.setLineDash([]);
        ctx.strokeStyle = rgba(colour, alpha);
        ctx.stroke(obs);
        if (anyEst) {
          ctx.setLineDash(ESTIMATED_TRAIL.dash);
          ctx.strokeStyle = rgba(colour, alpha * ESTIMATED_TRAIL.alphaScale);
          ctx.stroke(est);
          ctx.setLineDash([]);
        }
      }
    }
    // ball trail
    ctx.beginPath();
    let pen = false;
    for (let i = start; i <= end; i++) {
      const x = r.ball[i * 2] as number;
      const y = r.ball[i * 2 + 1] as number;
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        pen = false;
        continue;
      }
      const px = toCanvasX(tf, x);
      const py = toCanvasY(tf, y);
      if (pen) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
      pen = true;
    }
    ctx.setLineDash([2, 3]);
    ctx.lineWidth = Math.max(1, lw * 0.7);
    ctx.strokeStyle = rgba(this.theme.ball, 0.4);
    ctx.stroke();
    ctx.restore();
  }

  private drawnById(id: string | null): DrawnPlayer | null {
    if (!id) return null;
    for (const p of this.drawn) if (p.id === id) return p;
    return null;
  }

  private drawPassOptions(ctx: CanvasRenderingContext2D, tf: PitchTransform, a: FrameAnalysisOut): void {
    if (a.ball_x === null || a.ball_y === null || a.pass_options.length === 0) return;
    const s = this.scene;
    const diff = s.scenario && s.view === "difference";
    const deltas = new Map<string, number>();
    if (diff && s.scenario) for (const d of s.scenario.option_deltas) if (d.delta !== null) deltas.set(d.receiver_id, d.delta);
    const bx = toCanvasX(tf, a.ball_x);
    const by = toCanvasY(tf, a.ball_y);
    const opts = [...a.pass_options].sort((p, q) => primaryP(p) - primaryP(q));
    const k = Math.max(0.6, Math.min(1.3, tf.scale / 9));
    const top = new Set([...a.pass_options].sort((p, q) => primaryP(q) - primaryP(p)).slice(0, 3).map((o) => o.receiver_id));
    ctx.save();
    ctx.lineCap = "round";
    for (const o of opts) {
      const hl = s.highlightReceiver === o.receiver_id;
      const p = Math.max(0, Math.min(1, primaryP(o)));
      let colour = this.theme.text;
      let alpha = 0.16 + 0.7 * p;
      if (diff) {
        const d = deltas.get(o.receiver_id) ?? 0;
        colour = d > 0.005 ? this.theme.positive : d < -0.005 ? this.theme.negative : this.theme.textMuted;
        alpha = Math.abs(d) > 0.005 ? Math.min(1, 0.45 + Math.abs(d) * 3) : 0.25;
      }
      if (hl) {
        colour = this.theme.accent;
        alpha = 1;
      }
      const tx = toCanvasX(tf, o.target_x);
      const ty = toCanvasY(tf, o.target_y);
      ctx.strokeStyle = rgba(colour, alpha);
      ctx.lineWidth = (1 + 4.5 * p) * k + (hl ? 1.5 : 0);
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(tx, ty);
      ctx.stroke();
    }
    // probability tags for the strongest options (and the highlighted one)
    const R = this.radiusPx();
    if (R >= 6) {
      ctx.font = `600 ${Math.round(Math.max(10, Math.min(12, R * 1.05)))}px ${fontStack()}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const o of a.pass_options) {
        if (!top.has(o.receiver_id) && s.highlightReceiver !== o.receiver_id) continue;
        this.tag(ctx, toCanvasX(tf, o.target_x), toCanvasY(tf, o.target_y) - R - 10, this.optionTag(o, diff ? deltas.get(o.receiver_id) : undefined), s.highlightReceiver === o.receiver_id);
      }
    }
    ctx.restore();
  }

  private optionTag(o: PassOptionOut, delta?: number): string {
    const pct = `${Math.round(primaryP(o) * 100)}%`;
    if (delta === undefined) return pct;
    const d = Math.round(delta * 100);
    return `${pct} (${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d)})`;
  }

  private tag(ctx: CanvasRenderingContext2D, x: number, y: number, text: string, accent: boolean): void {
    const w = ctx.measureText(text).width + 10;
    const h = 17;
    ctx.fillStyle = accent ? rgba(this.theme.accent, 0.95) : "rgba(12, 15, 20, 0.82)";
    roundRect(ctx, x - w / 2, y - h / 2, w, h, 5);
    ctx.fill();
    ctx.fillStyle = accent ? this.theme.onTeam : this.theme.text;
    ctx.fillText(text, x, y + 0.5);
  }

  private drawPressure(ctx: CanvasRenderingContext2D, tf: PitchTransform, a: FrameAnalysisOut): void {
    const pr = a.pressure;
    const carrier = this.drawnById(a.carrier_id);
    if (!pr || !carrier) return;
    const cx = toCanvasX(tf, carrier.x);
    const cy = toCanvasY(tf, carrier.y);
    const d = pr.nearest_defender_distance_m;
    const colour = d < 2 ? this.theme.negative : d < 4 ? this.theme.warning : this.theme.textMuted;
    ctx.save();
    const def = this.drawnById(pr.nearest_defender_id);
    if (def) {
      const dx = toCanvasX(tf, def.x);
      const dy = toCanvasY(tf, def.y);
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = rgba(colour, 0.9);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(dx, dy);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    const ringR = Math.max(this.radiusPx() + 6, metresToPx(tf, 2.5));
    ctx.lineWidth = 2;
    ctx.strokeStyle = rgba(colour, 0.95);
    ctx.fillStyle = rgba(colour, 0.1);
    ctx.beginPath();
    ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  private drawGhosts(ctx: CanvasRenderingContext2D, tf: PitchTransform): void {
    const s = this.scene;
    const R = this.radiusPx();
    const moves: { id: string; fx: number; fy: number; tx: number; ty: number; prov: ProvenanceLabel }[] = [];
    const seen = new Set<string>();
    if (s.scenario) {
      for (const m of s.scenario.movements) {
        if (s.pending.has(m.player_id)) continue;
        moves.push({ id: m.player_id, fx: m.from_x, fy: m.from_y, tx: m.to_x, ty: m.to_y, prov: m.provenance });
        seen.add(m.player_id);
      }
    }
    const obsPos = (id: string): Vec2 | null => {
      const r = s.replay;
      if (!r) return null;
      const k = r.players.findIndex((p) => p.player_id === id);
      if (k < 0) return null;
      const v = { x: 0, y: 0 };
      return sampleXY(r.xy, r.nPlayers, r.nFrames, s.t, k, v, r.period) ? v : null;
    };
    for (const [id, pos] of s.pending) {
      const o = obsPos(id);
      if (o) moves.push({ id, fx: o.x, fy: o.y, tx: pos.x, ty: pos.y, prov: "hypothetical" });
      seen.add(id);
    }
    if (s.drag && !seen.has(s.drag.playerId)) {
      moves.push({ id: s.drag.playerId, fx: s.drag.fromX, fy: s.drag.fromY, tx: s.drag.x, ty: s.drag.y, prov: "hypothetical" });
    } else if (s.drag) {
      const m = moves.find((q) => q.id === s.drag?.playerId);
      if (m) {
        m.tx = s.drag.x;
        m.ty = s.drag.y;
      }
    }
    const players = s.replay?.players ?? [];
    ctx.save();
    for (const m of moves) {
      const pl = players.find((p) => p.player_id === m.id);
      const fx = toCanvasX(tf, m.fx);
      const fy = toCanvasY(tf, m.fy);
      const tx = toCanvasX(tf, m.tx);
      const ty = toCanvasY(tf, m.ty);
      // ghost at the observed position
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = rgba(pl ? this.teamColour(pl.team) : this.theme.text, 0.75);
      ctx.fillStyle = rgba(pl ? this.teamColour(pl.team) : this.theme.text, 0.1);
      ctx.beginPath();
      ctx.arc(fx, fy, R, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      if (pl && R >= 6) {
        ctx.setLineDash([]);
        ctx.fillStyle = rgba(this.theme.text, 0.6);
        ctx.font = `600 ${Math.round(R * 0.95)}px ${fontStack()}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(pl.jersey), fx, fy + 0.5);
      }
      // arrow from the observed to the new position
      const len = Math.hypot(tx - fx, ty - fy);
      if (len > R * 2) {
        const ux = (tx - fx) / len;
        const uy = (ty - fy) / len;
        const sx = fx + ux * (R + 2);
        const sy = fy + uy * (R + 2);
        const ex = tx - ux * (R + 4);
        const ey = ty - uy * (R + 4);
        // user moves: violet (hypothetical) long dashes; simulated defender responses: white
        // (simulated) short dashes plus a "sim" tag, so the two can never be confused
        const sim = m.prov === "simulated";
        const colour = sim ? this.theme.text : this.theme.accent;
        ctx.strokeStyle = rgba(colour, 0.9);
        ctx.fillStyle = rgba(colour, 0.9);
        ctx.lineWidth = 1.75;
        ctx.setLineDash(sim ? [2, 3] : [6, 4]);
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.setLineDash([]);
        const ah = 7;
        ctx.beginPath();
        ctx.moveTo(ex + ux * 2, ey + uy * 2);
        ctx.lineTo(ex - ux * ah - uy * ah * 0.55, ey - uy * ah + ux * ah * 0.55);
        ctx.lineTo(ex - ux * ah + uy * ah * 0.55, ey - uy * ah - ux * ah * 0.55);
        ctx.closePath();
        ctx.fill();
      }
      if (m.prov === "simulated") {
        // tag beside the ghost, on the side away from the arrow
        const off = R + 12;
        const lx = len > 0 ? fx - ((tx - fx) / len) * off : fx;
        const ly = len > 0 ? fy - ((ty - fy) / len) * off : fy - off;
        ctx.font = `600 10px ${fontStack()}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const w = ctx.measureText("sim").width + 8;
        ctx.fillStyle = "rgba(12, 15, 20, 0.85)";
        roundRect(ctx, lx - w / 2, ly - 7, w, 14, 4);
        ctx.fill();
        ctx.setLineDash([2, 2]);
        ctx.lineWidth = 1;
        ctx.strokeStyle = rgba(this.theme.text, 0.8);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = this.theme.text;
        ctx.fillText("sim", lx, ly + 0.5);
      }
    }
    ctx.restore();
  }

  private drawReach(ctx: CanvasRenderingContext2D, tf: PitchTransform, d: DragState): void {
    if (d.originX === null || d.originY === null || d.radius === null) return;
    const ox = toCanvasX(tf, d.originX);
    const oy = toCanvasY(tf, d.originY);
    const r = metresToPx(tf, d.radius);
    const colour = d.outside ? this.theme.negative : this.theme.accent;
    ctx.save();
    ctx.fillStyle = rgba(colour, d.outside ? 0.08 : 0.1);
    ctx.strokeStyle = rgba(colour, 0.9);
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.arc(ox, oy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);
    // origin marker (observed position horizon_s earlier)
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(ox - 4, oy - 4);
    ctx.lineTo(ox + 4, oy + 4);
    ctx.moveTo(ox + 4, oy - 4);
    ctx.lineTo(ox - 4, oy + 4);
    ctx.stroke();
    ctx.restore();
  }

  /** Halo under the players a focused explanation statement refers to (traceability). */
  private drawTrace(ctx: CanvasRenderingContext2D, tf: PitchTransform): void {
    const ids = new Set(this.scene.tracePlayers);
    const R = this.radiusPx();
    ctx.save();
    for (const p of this.drawn) {
      if (!ids.has(p.id)) continue;
      const x = toCanvasX(tf, p.x);
      const y = toCanvasY(tf, p.y);
      ctx.fillStyle = rgba(this.theme.accent, 0.22);
      ctx.strokeStyle = rgba(this.theme.accent, 0.95);
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(x, y, R + 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  private drawPlayers(ctx: CanvasRenderingContext2D, tf: PitchTransform, active: FrameAnalysisOut | null): void {
    const s = this.scene;
    const R = this.radiusPx();
    const showNumbers = R >= 6;
    ctx.save();
    ctx.font = `700 ${Math.round(R * 1.0)}px ${fontStack()}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const carrierId = active?.carrier_id ?? null;
    for (const p of this.drawn) {
      const x = toCanvasX(tf, p.x);
      const y = toCanvasY(tf, p.y);
      const colour = this.teamColour(p.team);
      const st = discStyle(styleKindFor(p.prov));
      // drop shadow (lighter under hollow estimated discs so they stay hollow)
      ctx.fillStyle = st.fillAlpha < 0.2 ? "rgba(0, 0, 0, 0.18)" : "rgba(0, 0, 0, 0.35)";
      ctx.beginPath();
      ctx.arc(x + 1, y + 1.5, R, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y, R, 0, Math.PI * 2);
      if (st.backingAlpha > 0) {
        ctx.fillStyle = rgba(this.theme.surround, st.backingAlpha);
        ctx.fill();
      }
      ctx.fillStyle = st.fillAlpha >= 1 ? colour : rgba(colour, st.fillAlpha);
      ctx.fill();
      const ringBase =
        st.ringColour === "team"
          ? colour
          : st.ringColour === "accent"
            ? this.theme.accent
            : st.ringColour === "text"
              ? this.theme.text
              : "#000000";
      ctx.lineWidth = st.ringWidth;
      ctx.strokeStyle = rgba(ringBase, st.ringAlpha);
      ctx.setLineDash(st.dash);
      ctx.lineCap = st.ring === "dotted" ? "round" : "butt";
      if (st.ring !== "outline") {
        // inset the dashed ring so the hollow disc keeps the same footprint
        ctx.beginPath();
        ctx.arc(x, y, R - st.ringWidth / 2 + 0.5, 0, Math.PI * 2);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      if (showNumbers) {
        const nc = st.numberColour === "onTeam" ? this.theme.onTeam : st.numberColour === "team" ? colour : this.theme.text;
        ctx.fillStyle = rgba(nc, st.numberAlpha);
        ctx.fillText(String(p.jersey), x, y + 0.5);
      }
      if (carrierId === p.id) {
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = rgba(this.theme.ball, 0.95);
        ctx.beginPath();
        ctx.arc(x, y, R + 2.5, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (s.selectedId === p.id) {
        ctx.lineWidth = 2;
        ctx.strokeStyle = this.theme.accent;
        ctx.beginPath();
        ctx.arc(x, y, R + 4.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  private drawBall(ctx: CanvasRenderingContext2D, tf: PitchTransform): void {
    if (!this.ball) return;
    const x = toCanvasX(tf, this.ball.x);
    const y = toCanvasY(tf, this.ball.y);
    const r = Math.max(3, Math.min(6, 0.42 * tf.scale));
    ctx.save();
    ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
    ctx.beginPath();
    ctx.arc(x + 1, y + 1.5, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = this.theme.ball;
    ctx.strokeStyle = "#0b0e12";
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}

function fontStack(): string {
  return 'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
