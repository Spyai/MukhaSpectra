import { useId, type ReactNode } from 'react';
import { Icon, type IconName } from './Icon';

/** Toggle chip (aria-pressed button) — big touch target, icon + text, never colour alone. */
export function ToggleChip({
  pressed,
  onChange,
  children,
  hint,
  icon,
  testId,
  size = 'md',
}: {
  pressed: boolean;
  onChange: (next: boolean) => void;
  children: ReactNode;
  hint?: string;
  icon?: IconName;
  testId?: string;
  size?: 'sm' | 'md';
}) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      onClick={() => onChange(!pressed)}
      data-testid={testId}
      className={`inline-flex items-center gap-2 rounded-xl border text-left text-sm font-semibold transition-colors ${
        size === 'sm' ? 'min-h-10 px-2.5 py-1.5' : 'min-h-11 px-3.5 py-2'
      } ${
        pressed
          ? 'border-violet bg-violet-soft text-fg'
          : 'border-line bg-surface-2 text-fg-muted hover:border-line-strong hover:text-fg'
      }`}
    >
      <span
        aria-hidden="true"
        className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-md border-2 ${
          pressed ? 'border-violet bg-violet text-on-violet' : 'border-line-strong'
        }`}
      >
        {pressed ? <Icon name="check" size={13} strokeWidth={3} /> : null}
      </span>
      {icon ? <Icon name={icon} size={16} className="shrink-0" /> : null}
      <span className="flex flex-col leading-tight">
        <span>{children}</span>
        {hint ? <span className="mt-0.5 text-xs font-normal text-fg-subtle">{hint}</span> : null}
      </span>
    </button>
  );
}

export function Segmented<T extends string>({
  label,
  hint,
  value,
  options,
  onChange,
  name,
  compact = false,
}: {
  label: string;
  hint?: string;
  value: T;
  /** `muted`: an "unknown / not asked" option, drawn dashed when selected. */
  options: Array<{ value: T; label: string; muted?: boolean }>;
  onChange: (v: T) => void;
  name?: string;
  compact?: boolean;
}) {
  const auto = useId();
  const group = name ?? auto;
  return (
    <fieldset>
      <legend className="field-label">
        {label}
        {hint ? (
          <span className="mt-0.5 block text-xs font-normal text-fg-subtle">{hint}</span>
        ) : null}
      </legend>
      <div className={`flex flex-wrap ${compact ? 'gap-1.5' : 'gap-2'}`}>
        {options.map((o) => (
          <label
            key={o.value}
            className={`relative inline-flex min-h-11 cursor-pointer items-center rounded-xl border text-sm font-semibold transition-colors ${compact ? 'px-3' : 'px-3.5'} has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-focus ${
              value === o.value
                ? o.muted
                  ? 'border-dashed border-line-strong bg-surface-2 text-fg'
                  : 'border-violet bg-violet-soft text-fg'
                : 'border-line bg-surface-2 text-fg-muted hover:border-line-strong'
            }`}
          >
            <input
              type="radio"
              name={group}
              value={o.value}
              checked={value === o.value}
              onChange={() => onChange(o.value)}
              className="sr-only"
            />
            {o.label}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export function Checkbox({
  checked,
  onChange,
  children,
  id,
  invalid,
  describedBy,
  required,
  testId,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  children: ReactNode;
  id?: string;
  invalid?: boolean;
  describedBy?: string;
  required?: boolean;
  testId?: string;
}) {
  const auto = useId();
  const inputId = id ?? auto;
  return (
    <label
      htmlFor={inputId}
      className={`relative flex cursor-pointer gap-3 rounded-xl border p-3.5 transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-focus ${
        invalid
          ? 'border-danger bg-risk-high-bg'
          : checked
            ? 'border-violet bg-violet-soft'
            : 'border-line bg-surface-2 hover:border-line-strong'
      }`}
    >
      <input
        id={inputId}
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        aria-invalid={invalid || undefined}
        aria-describedby={describedBy}
        aria-required={required || undefined}
        data-testid={testId}
        className="sr-only"
        // The input is visually hidden: bring the whole label clear of the sticky bars.
        onFocus={(e) => e.currentTarget.closest('label')?.scrollIntoView({ block: 'nearest' })}
      />
      <span
        aria-hidden="true"
        className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md border-2 ${
          checked
            ? 'border-violet bg-violet text-on-violet'
            : invalid
              ? 'border-danger'
              : 'border-line-strong'
        }`}
      >
        {checked ? <Icon name="check" size={15} strokeWidth={3} /> : null}
      </span>
      <span className="text-sm leading-relaxed text-fg">{children}</span>
    </label>
  );
}

export function FieldError({ id, children }: { id: string; children?: ReactNode }) {
  if (!children) return null;
  return (
    <p
      id={id}
      className="mt-1.5 flex items-start gap-1.5 text-sm font-medium text-danger"
      data-testid="field-error"
    >
      <Icon name="alert" size={16} className="mt-0.5 shrink-0" />
      <span>{children}</span>
    </p>
  );
}

export function Section({
  id,
  step,
  title,
  children,
}: {
  id: string;
  step: number;
  title: string;
  children: ReactNode;
}) {
  return (
    <section aria-labelledby={`${id}-title`} className="card p-5 sm:p-6">
      <h2 id={`${id}-title`} className="flex items-center gap-3 text-lg font-bold tracking-tight">
        <span className="flex h-8 w-8 items-center justify-center rounded-full bg-violet-soft text-sm font-bold text-violet-text tabular">
          {step}
        </span>
        {title}
      </h2>
      <div className="mt-5">{children}</div>
    </section>
  );
}
