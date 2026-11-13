/** Canvas palette, read from the CSS tokens so the DOM and the pitch never drift. */

export interface RGB {
  r: number;
  g: number;
  b: number;
}

export interface CanvasTheme {
  pitch: string;
  pitchStripe: string;
  pitchLine: string;
  surround: string;
  home: string;
  away: string;
  onTeam: string;
  accent: string;
  positive: string;
  negative: string;
  warning: string;
  shadow: string;
  ball: string;
  text: string;
  textMuted: string;
}

export const DEFAULT_THEME: CanvasTheme = {
  pitch: "#17241f",
  pitchStripe: "#1a2923",
  pitchLine: "rgba(226, 238, 232, 0.5)",
  surround: "#10171a",
  home: "#f0a24e",
  away: "#5ec6e4",
  onTeam: "#11161c",
  accent: "#b7a3ff",
  positive: "#5ad19d",
  negative: "#f2787a",
  warning: "#e9c46a",
  shadow: "#b8e35a",
  ball: "#f7f7f2",
  text: "#e8ebef",
  textMuted: "#a3adbb",
};

const VARS: Record<keyof CanvasTheme, string> = {
  pitch: "--c-pitch",
  pitchStripe: "--c-pitch-stripe",
  pitchLine: "--c-pitch-line",
  surround: "--c-pitch-surround",
  home: "--c-home",
  away: "--c-away",
  onTeam: "--c-on-team",
  accent: "--c-accent",
  positive: "--c-positive",
  negative: "--c-negative",
  warning: "--c-warning",
  shadow: "--c-shadow",
  ball: "--c-ball",
  text: "--c-text",
  textMuted: "--c-text-muted",
};

export function readTheme(el: Element = document.documentElement): CanvasTheme {
  const cs = getComputedStyle(el);
  const out = { ...DEFAULT_THEME };
  for (const k of Object.keys(VARS) as (keyof CanvasTheme)[]) {
    const v = cs.getPropertyValue(VARS[k]).trim();
    if (v) out[k] = v;
  }
  return out;
}

export function hexToRgb(hex: string): RGB {
  const h = hex.trim().replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h.slice(0, 6);
  const n = parseInt(full, 16);
  if (!Number.isFinite(n)) return { r: 255, g: 255, b: 255 };
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

export function rgba(hex: string, a: number): string {
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}
