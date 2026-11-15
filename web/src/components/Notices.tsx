/** API warnings, shown verbatim in a subtle notice area. */
export function Notices({ warnings, title = "Notes from the API" }: { warnings: string[]; title?: string }) {
  if (warnings.length === 0) return null;
  return (
    <div className="notices" role="note" aria-label={title} data-testid="api-warnings">
      <p className="notices-title">{title}</p>
      <ul>
        {warnings.map((w, i) => (
          <li key={i}>{w}</li>
        ))}
      </ul>
    </div>
  );
}
