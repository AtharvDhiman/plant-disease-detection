/**
 * Shared UI primitives.
 *
 * Kept in one module because they are small, stateless and always used
 * together; splitting them into a dozen files would add friction without
 * adding clarity.
 */
import clsx from 'clsx';
import { Info, Loader2, XCircle } from 'lucide-react';
import { useEffect, useState } from 'react';

/* ------------------------------------------------------------------- Card */

export function Card({ className, children, ...props }) {
  return (
    <div className={clsx('surface-card p-5', className)} {...props}>
      {children}
    </div>
  );
}

export function CardHeader({ title, subtitle, icon: Icon, action, className }) {
  return (
    <div className={clsx('flex items-start justify-between gap-4 mb-4', className)}>
      <div className="flex items-start gap-3 min-w-0">
        {Icon && (
          <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-leaf-100 text-leaf-700 dark:bg-leaf-900/60 dark:text-leaf-300">
            <Icon size={18} />
          </span>
        )}
        <div className="min-w-0">
          <h3 className="font-semibold leading-tight truncate">{title}</h3>
          {subtitle && <p className="text-sm text-secondary mt-0.5">{subtitle}</p>}
        </div>
      </div>
      {action}
    </div>
  );
}

/* ------------------------------------------------------------------ Button */

const BUTTON_VARIANTS = {
  primary:
    'bg-leaf-600 text-white hover:bg-leaf-700 active:bg-leaf-800 disabled:bg-leaf-600/50 shadow-sm',
  secondary:
    'bg-[var(--surface-raised)] text-[var(--text-primary)] border border-[var(--border-strong)] hover:bg-[var(--surface-sunken)]',
  ghost: 'text-[var(--text-secondary)] hover:bg-[var(--surface-sunken)]',
  danger: 'bg-clay-600 text-white hover:bg-clay-700 active:bg-clay-800 shadow-sm',
};

const BUTTON_SIZES = {
  sm: 'px-3 py-1.5 text-sm gap-1.5',
  md: 'px-4 py-2.5 text-sm gap-2',
  lg: 'px-6 py-3 text-base gap-2.5',
};

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  icon: Icon,
  className,
  children,
  disabled,
  ...props
}) {
  return (
    <button
      className={clsx(
        'inline-flex items-center justify-center rounded-lg font-medium transition-colors duration-150',
        'disabled:cursor-not-allowed disabled:opacity-60',
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? <Loader2 size={16} className="animate-spin" /> : Icon && <Icon size={16} />}
      {children}
    </button>
  );
}

/* -------------------------------------------------------------------- Chip */

export function Chip({ className, children, ...props }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium',
        className,
      )}
      {...props}
    >
      {children}
    </span>
  );
}

/* --------------------------------------------------------------- Skeletons */

export function Skeleton({ className }) {
  return <div className={clsx('skeleton rounded-md', className)} />;
}

export function SkeletonCard({ lines = 3 }) {
  return (
    <Card>
      <Skeleton className="h-5 w-1/3 mb-4" />
      {Array.from({ length: lines }).map((_, index) => (
        <Skeleton key={index} className={clsx('h-4 mb-2', index === lines - 1 ? 'w-2/3' : 'w-full')} />
      ))}
    </Card>
  );
}

export function SkeletonStats({ count = 4 }) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {Array.from({ length: count }).map((_, index) => (
        <Card key={index}>
          <Skeleton className="h-4 w-20 mb-3" />
          <Skeleton className="h-8 w-24" />
        </Card>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------ Empty / Error */

export function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center py-14 px-6 text-center">
      {Icon && (
        <span className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-[var(--surface-sunken)] text-[var(--text-muted)]">
          <Icon size={26} />
        </span>
      )}
      <h3 className="font-semibold text-lg">{title}</h3>
      {description && <p className="text-secondary text-sm mt-1.5 max-w-md">{description}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function ErrorState({ error, onRetry, title = 'Something went wrong' }) {
  return (
    <Card className="border-clay-300 dark:border-clay-800">
      <div className="flex items-start gap-3">
        <XCircle size={20} className="text-clay-600 shrink-0 mt-0.5" />
        <div className="min-w-0 flex-1">
          <h3 className="font-semibold">{title}</h3>
          <p className="text-secondary text-sm mt-1">{error?.message || 'Unknown error.'}</p>
          {error?.hint && <p className="text-muted text-sm mt-1">{error.hint}</p>}
          {onRetry && (
            <Button variant="secondary" size="sm" className="mt-3" onClick={onRetry}>
              Try again
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}

/* ------------------------------------------------------------------ Tabs */

export function Tabs({ tabs, active, onChange, className }) {
  return (
    <div
      role="tablist"
      className={clsx(
        'inline-flex gap-1 rounded-xl bg-[var(--surface-sunken)] p-1 overflow-x-auto max-w-full',
        className,
      )}
    >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          role="tab"
          aria-selected={active === tab.id}
          onClick={() => onChange(tab.id)}
          className={clsx(
            'whitespace-nowrap rounded-lg px-3.5 py-2 text-sm font-medium transition-colors',
            active === tab.id
              ? 'bg-[var(--surface-raised)] text-[var(--text-primary)] shadow-sm'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]',
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

/* --------------------------------------------------------------- StatCard */

export function StatCard({ label, value, sub, icon: Icon, tone = 'leaf', loading }) {
  const tones = {
    leaf: 'bg-leaf-100 text-leaf-700 dark:bg-leaf-900/60 dark:text-leaf-300',
    clay: 'bg-clay-100 text-clay-700 dark:bg-clay-900/60 dark:text-clay-300',
    amber: 'bg-amber-100 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300',
    neutral: 'bg-[var(--surface-sunken)] text-[var(--text-secondary)]',
  };
  return (
    <Card className="flex items-start justify-between gap-3">
      <div className="min-w-0">
        <p className="text-sm text-secondary">{label}</p>
        {loading ? (
          <Skeleton className="h-8 w-24 mt-2" />
        ) : (
          <p className="text-2xl font-semibold mt-1 tabular-nums truncate">{value}</p>
        )}
        {sub && <p className="text-xs text-muted mt-1 truncate">{sub}</p>}
      </div>
      {Icon && (
        <span className={clsx('flex h-10 w-10 shrink-0 items-center justify-center rounded-xl', tones[tone])}>
          <Icon size={19} />
        </span>
      )}
    </Card>
  );
}

/* ------------------------------------------------------------- ProgressBar */

export function ProgressBar({ value, className, barClassName, label }) {
  const clamped = Math.max(0, Math.min(1, value || 0));
  return (
    <div className={clsx('w-full', className)}>
      <div
        className="h-2 w-full overflow-hidden rounded-full bg-[var(--surface-sunken)]"
        role="progressbar"
        aria-valuenow={Math.round(clamped * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
      >
        <div
          className={clsx('h-full rounded-full transition-[width] duration-500', barClassName || 'bg-leaf-500')}
          style={{ width: `${clamped * 100}%` }}
        />
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- Tooltip */

export function InfoTooltip({ text }) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return undefined;
    const close = () => setOpen(false);
    window.addEventListener('scroll', close, true);
    return () => window.removeEventListener('scroll', close, true);
  }, [open]);

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        className="text-[var(--text-muted)] hover:text-[var(--text-primary)]"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
        aria-label="More information"
      >
        <Info size={14} />
      </button>
      {open && (
        <span className="absolute bottom-full left-1/2 z-30 mb-2 w-64 -translate-x-1/2 rounded-lg border border-[var(--border-strong)] bg-[var(--surface-raised)] p-2.5 text-xs font-normal leading-relaxed text-[var(--text-secondary)] shadow-lg">
          {text}
        </span>
      )}
    </span>
  );
}
