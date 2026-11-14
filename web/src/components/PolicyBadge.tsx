import type { PolicyInfo } from "../api/types";
import { POLICY_BADGE_TEXT, policyBadge } from "../lib/policies";

/**
 * The verdict badge of a defender response policy (see policyBadge). Renders nothing for the
 * static baseline. The tooltip says what the badge means; the API's own status text is shown
 * next to it by the callers that have room for it.
 */
export function PolicyBadge({
  info,
  testId,
}: {
  info: Pick<PolicyInfo, "name" | "validated" | "status"> | null | undefined;
  testId?: string;
}) {
  const kind = policyBadge(info);
  if (!kind) return null;
  return (
    <span
      className={`badge badge--policy badge--policy-${kind.replace(" ", "-")}`}
      data-testid={testId}
      data-badge={kind}
      title={POLICY_BADGE_TEXT[kind]}
    >
      {kind}
    </span>
  );
}
