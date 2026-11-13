import type { SourceInfo } from "../api/types";

/** Required attribution for the Metrica Sports sample data. Rendered from the API's `source`. */
export function SourceAttribution({ source, compact = false }: { source: SourceInfo; compact?: boolean }) {
  return (
    <p className={compact ? "attribution attribution--compact" : "attribution"} data-testid="source-attribution">
      <span className="attribution-label">Data</span> {source.name}: real, anonymised tracking and synchronised
      event data published by Metrica Sports (
      <a href={source.url} target="_blank" rel="noreferrer">
        {source.url.replace(/^https?:\/\//, "")}
      </a>
      ).{compact ? null : <span className="attribution-note"> {source.license_note}</span>}
    </p>
  );
}
