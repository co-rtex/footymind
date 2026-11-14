import type { ReactNode } from "react";
import { describeError } from "../lib/errors";

interface Props {
  kind: "loading" | "error" | "empty";
  title: string;
  /** plain text under the title */
  detail?: string;
  /**
   * The failed request. Its plain-language explanation replaces `detail`, the technical text is
   * shown small underneath, and Retry is offered only when retrying can help.
   */
  error?: Error | null;
  /** what was being loaded, for "We could not find <subject>" */
  subject?: string;
  onRetry?: () => void;
  /** extra action (for example a link back to the possession list) */
  action?: ReactNode;
  compact?: boolean;
}

export function StatusBlock({ kind, title, detail, error, subject, onRetry, action, compact = false }: Props) {
  const view = error ? describeError(error, subject) : null;
  const showRetry = onRetry && (!view || view.retryable);
  return (
    <div
      className={`status status--${kind}${compact ? " status--compact" : ""}`}
      role={kind === "error" ? "alert" : "status"}
      aria-live={kind === "error" ? "assertive" : "polite"}
    >
      {kind === "loading" ? <span className="spinner" aria-hidden="true" /> : null}
      <div className="status-body">
        <p className="status-title">{title}</p>
        {view ? <p className="status-detail">{view.message}</p> : detail ? <p className="status-detail">{detail}</p> : null}
        {view?.technical ? <p className="status-technical">{view.technical}</p> : null}
      </div>
      {showRetry || action ? (
        <div className="status-actions">
          {action}
          {showRetry ? (
            <button type="button" className="btn btn--ghost btn--sm" onClick={onRetry}>
              Retry
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
