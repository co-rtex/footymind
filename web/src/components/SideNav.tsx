import { revealSection } from "../lib/reveal";

/**
 * Jump links at the top of the analysis column, so the panels further down (similar moments,
 * events, the evidence of a scenario) are discoverable without scrolling to find them.
 */
export interface SideNavItem {
  /** element id to bring into view and focus (a heading with tabIndex -1, a control, or a <details>) */
  target: string;
  label: string;
}

export function SideNav({ items, label }: { items: SideNavItem[]; label: string }) {
  return (
    <nav className="side-nav" aria-label={label} data-testid="side-nav">
      <ul>
        {items.map((it) => (
          <li key={it.target}>
            {/* buttons, not #links: the location hash belongs to the router */}
            <button
              type="button"
              className="side-nav-link"
              onClick={() => revealSection(it.target, it.target === "policy-select" ? "center" : "start")}
            >
              {it.label}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
