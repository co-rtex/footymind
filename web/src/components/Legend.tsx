import { memo } from "react";
import type { Team } from "../api/types";
import type { ScenarioView } from "../render/renderer";
import { DEFAULT_TEAMS, type TeamNames } from "../lib/teams";
import { Disc, ProvenanceKey } from "./ProvenanceKey";

interface Props {
  attackingTeam: Team | null;
  showHeat: boolean;
  scenarioLayer: boolean;
  view: ScenarioView;
  showShadow?: boolean;
  /** the replay contains provider-extrapolated (off-camera) positions */
  showEstimated?: boolean;
  /** the ball is a provider extrapolation somewhere in the replay or at the shown frame */
  showEstimatedBall?: boolean;
  teams?: TeamNames;
}

export const Legend = memo(function Legend({
  attackingTeam,
  showHeat,
  scenarioLayer,
  view,
  showShadow = false,
  showEstimated = false,
  showEstimatedBall = false,
  teams = DEFAULT_TEAMS,
}: Props) {
  const att = attackingTeam === "away" ? "var(--c-away)" : "var(--c-home)";
  const def = attackingTeam === "away" ? "var(--c-home)" : "var(--c-away)";
  return (
    <div className="legend" aria-label="Legend" role="group">
      <ul className="legend-row">
        <li>
          <Disc fill="var(--c-home)" /> {teams.home}
        </li>
        <li>
          <Disc fill="var(--c-away)" /> {teams.away}
        </li>
        <li>
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <circle cx="7" cy="7" r="4" fill="var(--c-ball)" stroke="#0b0e12" strokeWidth="1.2" />
          </svg>
          Ball
        </li>
        {showEstimatedBall ? (
          <li data-testid="legend-ball-estimated">
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
              <circle cx="7" cy="7" r="4.2" fill="var(--c-ball)" fillOpacity="0.12" stroke="var(--c-ball)" strokeWidth="1.3" strokeDasharray="2 2" />
            </svg>
            Ball, estimated (provider extrapolation, tagged &ldquo;est.&rdquo;)
          </li>
        ) : null}
        {showEstimated ? (
          <>
            <li>
              <Disc fill="var(--c-text-muted)" /> Observed (solid)
            </li>
            <li data-testid="legend-estimated">
              <Disc fill="var(--c-text-muted)" opacity={0.1} stroke="var(--c-text-muted)" dash="2.6 2" /> Estimated (provider
              extrapolation, off camera)
            </li>
          </>
        ) : null}
        {scenarioLayer ? (
          <>
            {showEstimated ? null : (
              <li>
                <Disc fill="var(--c-text-muted)" /> Observed (solid)
              </li>
            )}
            <li>
              <Disc fill="var(--c-text-muted)" opacity={0.3} stroke="var(--c-accent)" dash="3.5 2.2" /> Hypothetical (you moved)
            </li>
            <li>
              <Disc fill="var(--c-text-muted)" opacity={0.3} stroke="var(--c-text)" dash="1 2.2" /> Simulated (defender
              response model; arrow labelled &ldquo;sim&rdquo;)
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
          Pass option (width and opacity: option-model probability)
        </li>
        <li>
          <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
            <circle cx="9" cy="9" r="7" fill="var(--c-negative)" fillOpacity="0.12" stroke="var(--c-negative)" strokeWidth="1.6" />
          </svg>
          Pressure ring
        </li>
        {showShadow ? (
          <li>
            <svg width="22" height="18" viewBox="0 0 22 18" aria-hidden="true">
              <defs>
                <radialGradient id="legend-shadow">
                  <stop offset="0" stopColor="var(--c-shadow)" stopOpacity="0.5" />
                  <stop offset="1" stopColor="var(--c-shadow)" stopOpacity="0" />
                </radialGradient>
              </defs>
              <circle cx="9" cy="9" r="8" fill="url(#legend-shadow)" stroke="var(--c-shadow)" strokeWidth="1.2" />
              <circle cx="9" cy="9" r="3.5" fill="var(--c-text-muted)" />
            </svg>
            Defensive shadow (experimental): halo size and opacity = estimated passing value removed; dashed line to the
            most suppressed receiver
          </li>
        ) : null}
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
      <details className="legend-more" id="labels-key" data-testid="legend-labels">
        <summary>What do observed, derived, estimated, hypothetical and simulated mean?</summary>
        <ProvenanceKey compact />
      </details>
    </div>
  );
});

/**
 * Compact colour key drawn over the pitch's bottom margin, between the attack-direction labels,
 * so the control colours are explained without scrolling to the full legend (audit F10).
 * Hidden on narrow screens, where the full legend sits just below the transport.
 */
export const PitchKey = memo(function PitchKey({ attackingTeam, view }: { attackingTeam: Team | null; view: ScenarioView }) {
  const att = attackingTeam === "away" ? "var(--c-away)" : "var(--c-home)";
  const def = attackingTeam === "away" ? "var(--c-home)" : "var(--c-away)";
  const diff = view === "difference";
  return (
    <div className="pitch-key" data-testid="pitch-key" aria-hidden="true">
      <span className="pitch-key-title">{diff ? "Δ est. control" : "Est. control"}</span>
      <span>{diff ? "attackers lose" : "defending"}</span>
      <span
        className="gradient gradient--sm"
        style={{
          background: diff
            ? "linear-gradient(90deg, var(--c-negative), transparent, var(--c-positive))"
            : `linear-gradient(90deg, ${def}, transparent, ${att})`,
        }}
      />
      <span>{diff ? "gain" : "attacking"}</span>
    </div>
  );
});
