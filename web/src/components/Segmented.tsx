import { useId, type KeyboardEvent } from "react";

interface Option<T extends string> {
  value: T;
  label: string;
  title?: string;
}

interface Props<T extends string> {
  label: string;
  value: T;
  options: Option<T>[];
  onChange: (v: T) => void;
  disabled?: boolean;
  hideLabel?: boolean;
  testId?: string;
}

/** Accessible segmented control (radio group with roving arrow-key selection). */
export function Segmented<T extends string>({ label, value, options, onChange, disabled, hideLabel, testId }: Props<T>) {
  const id = useId();
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;
    const i = options.findIndex((o) => o.value === value);
    let j = -1;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % options.length;
    if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i - 1 + options.length) % options.length;
    if (j >= 0) {
      e.preventDefault();
      e.stopPropagation();
      const next = options[j];
      if (next) {
        onChange(next.value);
        const btn = e.currentTarget.querySelectorAll<HTMLButtonElement>("button")[j];
        btn?.focus();
      }
    }
  };
  return (
    <div className="segmented-wrap">
      <span id={id} className={hideLabel ? "sr-only" : "segmented-label"}>
        {label}
      </span>
      <div className="segmented" role="radiogroup" aria-labelledby={id} onKeyDown={onKey} data-testid={testId}>
        {options.map((o) => {
          const on = o.value === value;
          return (
            <button
              key={o.value}
              type="button"
              role="radio"
              aria-checked={on}
              tabIndex={on ? 0 : -1}
              title={o.title}
              disabled={disabled}
              className={on ? "is-on" : undefined}
              onClick={() => onChange(o.value)}
            >
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
