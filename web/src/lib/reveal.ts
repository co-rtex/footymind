/**
 * Bring a section into view and move focus to it: a heading (tabIndex -1), a form control, or a
 * <details> (opened first, focus on its summary). Used by in-page jump links and the guided tour;
 * the location hash is the router's, so these never navigate.
 */
export function revealSection(id: string, block: ScrollLogicalPosition = "start"): void {
  const el = document.getElementById(id);
  if (!el) return;
  if (el instanceof HTMLDetailsElement) el.open = true;
  el.scrollIntoView({ block });
  const focusable = el instanceof HTMLDetailsElement ? el.querySelector("summary") : el;
  (focusable as HTMLElement | null)?.focus({ preventScroll: true });
}
