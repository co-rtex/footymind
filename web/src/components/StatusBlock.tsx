interface Props {
  kind: "loading" | "error" | "empty";
  title: string;
  detail?: string;
  onRetry?: () => void;
  compact?: boolean;
}

export function StatusBlock({ kind, title, detail, onRetry, compact = false }: Props) {
  return (
    <div
      className={`status status--${kind}${compact ? " status--compact" : ""}`}
      role={kind === "error" ? "alert" : "status"}
      aria-live={kind === "error" ? "assertive" : "polite"}
    >
      {kind === "loading" ? <span className="spinner" aria-hidden="true" /> : null}
      <div>
        <p className="status-title">{title}</p>
        {detail ? <p className="status-detail">{detail}</p> : null}
      </div>
      {onRetry ? (
        <button type="button" className="btn btn--ghost btn--sm" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}
