/** Small formatting helpers shared across the UI. */

/**
 * Format a 0-1 probability as a percentage.
 *
 * Values just under 1 must not round up to a flat "100%": a classifier is never
 * certain, and displaying certainty it does not have is the kind of small
 * dishonesty that erodes trust in the whole result. Anything in (0, 1) that
 * would round to the extremes is shown as a bound instead.
 */
export const percent = (value, digits = 1) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const scaled = value * 100;
  const rounded = Number(scaled.toFixed(digits));
  if (value < 1 && rounded >= 100) return `>${(100 - 10 ** -digits).toFixed(digits)}%`;
  if (value > 0 && rounded <= 0) return `<${(10 ** -digits).toFixed(digits)}%`;
  return `${scaled.toFixed(digits)}%`;
};

export const number = (value, digits = 0) =>
  value === null || value === undefined || Number.isNaN(value)
    ? '—'
    : Number(value).toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });

export const decimal = (value, digits = 4) =>
  value === null || value === undefined || Number.isNaN(value) ? '—' : Number(value).toFixed(digits);

// One decimal place is right for the 2-30 ms range the deep models occupy, but
// it renders the classical models' sub-0.05 ms latency as a flat "0.0 ms",
// which reads as a broken measurement rather than a very fast one. Sub-0.1 ms
// values therefore get the precision they need to stay meaningful.
export const milliseconds = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const ms = Number(value);
  if (ms === 0) return '0 ms';
  if (ms < 0.1) return `${ms.toFixed(3)} ms`;
  if (ms < 1) return `${ms.toFixed(2)} ms`;
  return `${ms.toFixed(1)} ms`;
};

export const megabytes = (value) =>
  value === null || value === undefined ? '—' : `${Number(value).toFixed(1)} MB`;

export const bytes = (value) => {
  if (value === null || value === undefined) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = Number(value);
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
};

export const duration = (seconds) => {
  if (seconds === null || seconds === undefined) return '—';
  const total = Math.round(Number(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes}m ${total % 60}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
};

export const parameters = (value) => {
  if (!value) return '—';
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(value);
};

/** `Tomato___Late_blight` -> `Tomato / Late blight`. */
export const prettyClass = (raw) =>
  (raw || '').replace('___', ' / ').replace(/_/g, ' ').trim();

export const formatDate = (value) => {
  if (!value) return '—';
  const date = new Date(value);
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const relativeTime = (value) => {
  if (!value) return '—';
  const then = new Date(value).getTime();
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
  return formatDate(value);
};

/* ------------------------------------------------------------------ colours */

export const CONFIDENCE_STYLES = {
  high: {
    label: 'High confidence',
    chip: 'bg-leaf-100 text-leaf-800 border-leaf-300 dark:bg-leaf-900/60 dark:text-leaf-200 dark:border-leaf-700',
    bar: 'bg-leaf-500',
    dot: 'bg-leaf-500',
  },
  medium: {
    label: 'Medium confidence',
    chip: 'bg-amber-100 text-amber-900 border-amber-300 dark:bg-amber-950/60 dark:text-amber-200 dark:border-amber-800',
    bar: 'bg-amber-500',
    dot: 'bg-amber-500',
  },
  low: {
    label: 'Low confidence',
    chip: 'bg-clay-100 text-clay-800 border-clay-300 dark:bg-clay-900/50 dark:text-clay-200 dark:border-clay-700',
    bar: 'bg-clay-500',
    dot: 'bg-clay-500',
  },
};

export const SEVERITY_STYLES = {
  none: 'bg-leaf-100 text-leaf-800 border-leaf-300 dark:bg-leaf-900/50 dark:text-leaf-200 dark:border-leaf-800',
  moderate:
    'bg-amber-100 text-amber-900 border-amber-300 dark:bg-amber-950/50 dark:text-amber-200 dark:border-amber-800',
  high: 'bg-orange-100 text-orange-900 border-orange-300 dark:bg-orange-950/50 dark:text-orange-200 dark:border-orange-800',
  critical:
    'bg-clay-100 text-clay-800 border-clay-400 dark:bg-clay-900/60 dark:text-clay-200 dark:border-clay-700',
  unknown: 'bg-neutral-100 text-neutral-700 border-neutral-300 dark:bg-neutral-800 dark:text-neutral-300 dark:border-neutral-700',
};

export const CATEGORY_STYLES = {
  fungal: 'bg-purple-100 text-purple-800 dark:bg-purple-950/50 dark:text-purple-200',
  bacterial: 'bg-sky-100 text-sky-800 dark:bg-sky-950/50 dark:text-sky-200',
  viral: 'bg-rose-100 text-rose-800 dark:bg-rose-950/50 dark:text-rose-200',
  oomycete: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-950/50 dark:text-indigo-200',
  pest: 'bg-orange-100 text-orange-800 dark:bg-orange-950/50 dark:text-orange-200',
  healthy: 'bg-leaf-100 text-leaf-800 dark:bg-leaf-900/50 dark:text-leaf-200',
  unknown: 'bg-neutral-100 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300',
};

export const STATUS_STYLES = {
  ok: { label: 'Analysed', tone: 'leaf' },
  low_confidence: { label: 'Low confidence', tone: 'clay' },
  out_of_distribution: { label: 'Not identified', tone: 'clay' },
  poor_quality: { label: 'Poor image quality', tone: 'amber' },
};

/** Categorical palette for charts; kept in one place so every chart agrees. */
export const CHART_COLORS = [
  '#2f7a4d',
  '#c1553b',
  '#d9a441',
  '#4b7ea8',
  '#8a5fa8',
  '#5faa71',
  '#de7a5e',
  '#6b8f3a',
  '#a03d29',
  '#3f8c54',
];
