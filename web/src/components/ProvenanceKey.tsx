import type { ReactNode } from "react";
import type { ProvenanceLabel } from "../api/types";
import { ProvBadge } from "./ProvBadge";

/** Player disc as drawn on the pitch (shared by the legend and the provenance key). */
export const Disc = ({
  fill,
  stroke,
  dash,
  opacity = 1,
}: {
  fill: string;
  stroke?: string;
  dash?: string;
  opacity?: number;
}) => (
  <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
    <circle
      cx="9"
      cy="9"
      r="6.5"
      fill={fill}
      fillOpacity={opacity}
      stroke={stroke ?? "rgba(0,0,0,.45)"}
      strokeWidth={stroke ? 1.8 : 1}
      strokeDasharray={dash}
      strokeLinecap="round"
    />
  </svg>
);

const HeatSwatch = () => (
  <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
    <defs>
      <linearGradient id="pk-heat" x1="0" x2="1">
        <stop offset="0" stopColor="var(--c-home)" stopOpacity="0.55" />
        <stop offset="1" stopColor="var(--c-away)" stopOpacity="0.55" />
      </linearGradient>
    </defs>
    <rect x="1.5" y="3" width="15" height="12" rx="3" fill="url(#pk-heat)" stroke="var(--c-warning)" strokeDasharray="1.5 1.5" />
  </svg>
);

const RuleSwatch = () => (
  <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
    <path d="M3 13 L9 5 L15 13" fill="none" stroke="var(--c-text-muted)" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);

interface Row {
  prov: ProvenanceLabel;
  glyph: ReactNode;
  what: string;
  example: string;
}

const ROWS: Row[] = [
  {
    prov: "observed",
    glyph: <Disc fill="var(--c-home)" />,
    what: "Measured by the tracking or event provider.",
    example: "Solid players, the ball, the events list, and the outcomes of similar moments.",
  },
  {
    prov: "derived",
    glyph: <RuleSwatch />,
    what: "Computed by a fixed rule from observed data, with no fitted model.",
    example: "Distances, speeds, pressure counts, time on the ball.",
  },
  {
    prov: "estimated",
    glyph: <HeatSwatch />,
    what: "A model output or a provider extrapolation. Useful, but not a measurement.",
    example: "Pitch control, pass-completion estimates, and hollow off-camera players on broadcast tracking.",
  },
  {
    prov: "hypothetical",
    glyph: <Disc fill="var(--c-home)" opacity={0.3} stroke="var(--c-accent)" dash="3.5 2.2" />,
    what: "A position you set in a what-if scenario. It did not happen.",
    example: "Violet dashed rings and arrows.",
  },
  {
    prov: "simulated",
    glyph: <Disc fill="var(--c-home)" opacity={0.3} stroke="var(--c-text)" dash="1 2.2" />,
    what: "Moved by a defender-response model in reaction to your edit. It did not happen either.",
    example: "White dotted rings and arrows tagged “sim”.",
  },
];

/**
 * The five provenance labels, with the pitch glyph and text badge each one uses. This is the
 * key to the app's visual language: everything on screen carries one of these labels.
 */
export function ProvenanceKey({ compact = false, headingId }: { compact?: boolean; headingId?: string }) {
  return (
    <dl className={`prov-key${compact ? " prov-key--compact" : ""}`} aria-labelledby={headingId} data-testid="provenance-key">
      {ROWS.map((r) => (
        <div key={r.prov} className="prov-key-row">
          <dt>
            <span className="prov-key-glyph">{r.glyph}</span>
            <ProvBadge prov={r.prov} />
          </dt>
          <dd>
            <span>{r.what}</span>
            {compact ? null : <span className="prov-key-example">{r.example}</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}
