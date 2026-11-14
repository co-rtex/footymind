import type { StatementOut } from "../api/types";

export interface StatementGroup {
  kind: string;
  title: string;
  /** shown when the group has no statement at this frame, so the layout stays put */
  empty: string;
  items: StatementOut[];
}

/** Fixed order and plain-language titles. Caveats come last and are always shown. */
const GROUPS: { kind: string; title: string; empty: string }[] = [
  { kind: "pressure", title: "Pressure on the ball", empty: "Nothing notable about pressure at this frame." },
  { kind: "trend", title: "In the last second", empty: "No notable change over the last second." },
  { kind: "option", title: "Passing options", empty: "No passing option stands out at this frame." },
  { kind: "caveat", title: "Caveats", empty: "No caveats reported for this frame." },
];

/**
 * Group statements by kind, keeping the backend's order within a group. The four known groups
 * are always present (possibly empty) so the panel does not jump between frames; statements
 * of an unknown kind are kept, in an "Other" group before the caveats.
 */
export function groupStatements(statements: StatementOut[]): StatementGroup[] {
  const known = new Set(GROUPS.map((g) => g.kind));
  const groups = GROUPS.map((g) => ({ ...g, items: statements.filter((s) => s.kind === g.kind) }));
  const other = statements.filter((s) => !known.has(s.kind));
  if (other.length > 0) {
    groups.splice(groups.length - 1, 0, { kind: "other", title: "Other", empty: "", items: other });
  }
  return groups;
}
