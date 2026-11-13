import type { SourceInfo } from "../api/types";
import { providerInfo } from "../lib/providers";

/** Required provider attribution (Metrica Sports, SkillCorner, ...), rendered from the API's `source`. */
export function SourceAttribution({ source, compact = false }: { source: SourceInfo; compact?: boolean }) {
  const info = providerInfo(source);
  return (
    <p className={compact ? "attribution attribution--compact" : "attribution"} data-testid="source-attribution">
      <span className="attribution-label">Data</span> {source.name}: {info.attribution} (
      <a href={source.url} target="_blank" rel="noreferrer">
        {source.url.replace(/^https?:\/\//, "")}
      </a>
      ).{compact ? null : <span className="attribution-note"> {source.license_note}</span>}
    </p>
  );
}
