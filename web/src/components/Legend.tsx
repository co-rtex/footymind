import type { Team } from "../api/types";
import type { ScenarioView } from "../render/renderer";

interface Props {
  attackingTeam: Team | null;
  showHeat: boolean;
  scenarioLayer: boolean;
  view: ScenarioView;
}

const Disc = ({ fill, stroke, dash, opacity = 1 }: { fill: string; stroke?: string; dash?: string; opacity?: number }) => (
  <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
    <circle cx="9" cy="9" r="6.5" fill={fill} fillOpacity={opacity} stroke={stroke ?? "rgba(0,0,0,.45)"} strokeWidth={stroke ? 1.8 : 1} strokeDasharray={dash} strokeLinecap="round" />
  </svg>
);

export function Legend({ attackingTeam, showHeat, scenarioLayer, view }: Props) {
  const att = attackingTeam === "away" ? "var(--c-away)" : "var(--c-home)";
  const def = attackingTeam === "away" ? "var(--c-home)" : "var(--c-away)";
  return (
    <div className="legend" aria-label="Legend" role="group">
      <ul className="legend-row">
        <li>
          <Disc fill="var(--c-home)" /> Home
        </li>
        <li>
          <Disc fill="var(--c-away)" /> Away
        </li>
        <li>
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <circle cx="7" cy="7" r="4" fill="var(--c-ball)" stroke="#0b0e12" strokeWidth="1.2" />
          </svg>
          Ball
        </li>
        {scenarioLayer ? (
          <>
            <li>
              <Disc fill="var(--c-home)" /> Observed (solid)
            </li>
            <li>
              <Disc fill="var(--c-home)" opacity={0.3} stroke="var(--c-accent)" dash="3.5 2.2" /> Hypothetical (you moved)
            </li>
            <li>
              <Disc fill="var(--c-home)" opacity={0.3} stroke="var(--c-text)" dash="1 2.2" /> Simulated (rule response)
            </li>
            <li>
              <Disc fill="transparent" stroke="var(--c-text-muted)" dash="2.5 2.5" /> Ghost: observed position
            </li>
          </>
        ) : null}
        <li>
          <svg width="28" height="12" viewBox="0 0 28 12" aria-hidden="true">
            <line x1="2" y1="9" x2="26" y2="3" stroke="var(--c-text)" strokeOpacity="0.85" strokeWidth="3.5" strokeLinecap="round" />
          </svg>
          Pass option (width and opacity: est. P(success))
        </li>
        <li>
          <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
            <circle cx="9" cy="9" r="7" fill="var(--c-negative)" fillOpacity="0.12" stroke="var(--c-negative)" strokeWidth="1.6" />
          </svg>
          Pressure ring
        </li>
      </ul>
      {showHeat ? (
        view === "difference" ? (
          <div className="legend-heat">
            <span className="legend-heat-title">Δ estimated pitch control (hypothetical − observed), model, uncalibrated</span>
            <span className="legend-heat-scale">
              <span>attackers lose</span>
              <span className="gradient" style={{ background: "linear-gradient(90deg, var(--c-negative), transparent, var(--c-positive))" }} />
              <span>attackers gain</span>
            </span>
          </div>
        ) : (
          <div className="legend-heat">
            <span className="legend-heat-title">Estimated pitch control (model, uncalibrated)</span>
            <span className="legend-heat-scale">
              <span>defending</span>
              <span className="gradient" style={{ background: `linear-gradient(90deg, ${def}, transparent, ${att})` }} />
              <span>attacking</span>
            </span>
          </div>
        )
      ) : null}
    </div>
  );
}
