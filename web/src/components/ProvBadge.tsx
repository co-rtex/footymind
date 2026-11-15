import type { ProvenanceLabel } from "../api/types";
import { PROVENANCE_TEXT } from "../lib/format";

/** Small label that makes the provenance of a value explicit and visually distinct. */
export function ProvBadge({ prov, children }: { prov: ProvenanceLabel; children?: string }) {
  return (
    <span className={`prov prov--${prov}`} title={PROVENANCE_TEXT[prov]}>
      {children ?? prov}
    </span>
  );
}
