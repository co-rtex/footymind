import type { FeaturedScenario } from "../lib/featured";
import type { TeamNames } from "../lib/teams";

export type TourStep = 1 | 2 | 3;
/** Sections the guide can bring into view: "compare" is the static-vs-response comparison. */
export type TourJump = "why" | "evidence" | "similar" | "response" | "result" | "compare";

interface Props {
  featured: FeaturedScenario;
  teams: TeamNames;
  step: TourStep;
  playing: boolean;
  /** label of the player the tour moves, e.g. "#9 Home" */
  player: string;
  /** label of the defender response step 3 compares with static defenders; null for static only */
  response: string | null;
  /** step 3: the featured move has been applied and its result is on screen */
  moved: boolean;
  onStep: (s: TourStep) => void;
  onWatch: () => void;
  onJump: (target: TourJump) => void;
  onExit: () => void;
}

const STEPS: { n: TourStep; label: string }[] = [
  { n: 1, label: "Watch" },
  { n: 2, label: "Understand" },
  { n: 3, label: "What if?" },
];

/**
 * Three-step guided path through a featured situation: watch the observed possession, read why
 * the moment matters, then apply the curated move and compare. Each step only drives the lab's
 * own controls (play, pause on the decision frame, apply the move), so everything on screen is
 * the normal, labelled lab output.
 */
export function GuideBar({ featured, teams, step, playing, player, response, moved, onStep, onWatch, onJump, onExit }: Props) {
  const text = featured.steps(teams);
  return (
    <section className="guide" aria-labelledby="guide-title" data-testid="guide" data-step={step}>
      <div className="guide-head">
        <div>
          <p className="eyebrow guide-eyebrow">Guided tour</p>
          <p id="guide-title" className="guide-title">
            {featured.title}
          </p>
        </div>
      </div>
      <ol className="guide-steps">
        {STEPS.map((s) => (
          <li key={s.n}>
            <button
              type="button"
              className={`guide-step${s.n === step ? " is-current" : s.n < step ? " is-done" : ""}`}
              aria-current={s.n === step ? "step" : undefined}
              onClick={() => onStep(s.n)}
              data-testid={`guide-step-${s.n}`}
            >
              <span className="step-n" aria-hidden="true">
                {s.n}
              </span>
              {s.label}
            </button>
          </li>
        ))}
      </ol>
      <div className="guide-body" aria-live="polite">
        {step === 1 ? (
          <>
            <p>{text.watch}</p>
            <div className="guide-actions">
              <button type="button" className="btn btn--primary btn--sm" onClick={onWatch} data-testid="guide-watch">
                {playing ? "Playing…" : "Play the possession"}
              </button>
              <button type="button" className="btn btn--sm" onClick={() => onStep(2)} data-testid="guide-next">
                Next: why this moment <span aria-hidden="true">→</span>
              </button>
            </div>
          </>
        ) : step === 2 ? (
          <>
            <p>
              {text.why} The colours are model estimates; the distances are derived from the tracking.
            </p>
            <div className="guide-actions">
              <button type="button" className="btn btn--sm" onClick={() => onJump("why")}>
                Why this moment
              </button>
              <button type="button" className="btn btn--sm" onClick={() => onJump("evidence")}>
                Pass options
              </button>
              <button type="button" className="btn btn--sm" onClick={() => onJump("similar")}>
                Similar moments
              </button>
              <button type="button" className="btn btn--primary btn--sm" onClick={() => onStep(3)} data-testid="guide-next">
                Next: move {player} <span aria-hidden="true">→</span>
              </button>
            </div>
          </>
        ) : (
          <>
            <p>
              {text.move}{" "}
              {moved && response ? (
                <span data-testid="guide-response">
                  The <strong>Defender response</strong> is set to <strong>{response}</strong>; the defender it moves is
                  simulated, not observed. <strong>Does the defender response matter?</strong> puts the static and response
                  results side by side. Choose <strong>Static (no response)</strong> to see the defenders stand still.
                </span>
              ) : moved ? (
                <>
                  Switch <strong>Observed</strong> / <strong>Hypothetical</strong> / <strong>Difference</strong> above the
                  pitch. The <strong>Defender response</strong> models react to a moved attacker or ball carrier: drag an
                  attacker to see one respond.
                </>
              ) : null}
            </p>
            <div className="guide-actions">
              <button type="button" className="btn btn--sm" onClick={() => onJump("result")}>
                Result
              </button>
              {response ? (
                <button type="button" className="btn btn--sm" onClick={() => onJump("compare")} data-testid="guide-compare">
                  Static vs response
                </button>
              ) : null}
              <button type="button" className="btn btn--sm" onClick={() => onJump("response")}>
                Defender response
              </button>
              <button type="button" className="btn btn--primary btn--sm" onClick={onExit} data-testid="guide-finish">
                Finish the tour
              </button>
            </div>
          </>
        )}
      </div>
      {/* last in tab order, shown top right */}
      <button type="button" className="btn btn--ghost btn--xs guide-exit" onClick={onExit} data-testid="guide-exit">
        Exit tour
      </button>
    </section>
  );
}
