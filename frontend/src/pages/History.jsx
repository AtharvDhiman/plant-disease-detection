import clsx from 'clsx';
import {
  ArrowLeft,
  Clock,
  Filter,
  History as HistoryIcon,
  ScanLine,
  Search,
  Trash2,
} from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';

import DiseaseInfoPanel from '../components/DiseaseInfoPanel';
import {
  Button,
  Card,
  CardHeader,
  Chip,
  EmptyState,
  ErrorState,
  Skeleton,
  SkeletonCard,
} from '../components/ui';
import { useToast } from '../components/ui/toast-context';
import { useApi } from '../hooks/useApi';
import { api, imageUrl } from '../services/api';
import {
  CONFIDENCE_STYLES,
  formatDate,
  milliseconds,
  percent,
  prettyClass,
  relativeTime,
} from '../utils/format';

const PAGE_SIZE = 12;

const SORTS = [
  { value: 'created_desc', label: 'Newest first' },
  { value: 'created_asc', label: 'Oldest first' },
  { value: 'confidence_desc', label: 'Highest confidence' },
  { value: 'confidence_asc', label: 'Lowest confidence' },
];

// A refused analysis must not look like a diagnosis in the list. The detail page
// already says why a prediction was rejected; without this the card showed the
// class and a confidence percentage for an image the system explicitly declined
// to identify, which is the opposite of what the rest of the pipeline promises.
const STATUS_BADGES = {
  ok: null,
  low_confidence: {
    label: 'Low confidence',
    className:
      'border-amber-300 bg-amber-50 text-amber-900 '
      + 'dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-100',
  },
  out_of_distribution: {
    label: 'Not identified',
    className:
      'border-clay-300 bg-clay-50 text-clay-900 '
      + 'dark:border-clay-800 dark:bg-clay-950/60 dark:text-clay-100',
  },
  poor_quality: {
    label: 'Poor image quality',
    className:
      'border-clay-300 bg-clay-50 text-clay-900 '
      + 'dark:border-clay-800 dark:bg-clay-950/60 dark:text-clay-100',
  },
  not_a_plant: {
    label: 'Not a plant leaf',
    className:
      'border-red-300 bg-red-50 text-red-900 '
      + 'dark:border-red-800 dark:bg-red-950/60 dark:text-red-100',
  },
};

const STATUSES = [
  { value: '', label: 'Any status' },
  { value: 'ok', label: 'Analysed' },
  { value: 'low_confidence', label: 'Low confidence' },
  { value: 'out_of_distribution', label: 'Not identified' },
  { value: 'poor_quality', label: 'Poor image quality' },
  { value: 'not_a_plant', label: 'Not a plant leaf' },
];

export default function History() {
  const [search, setSearch] = useState('');
  const [debounced, setDebounced] = useState('');
  const [sort, setSort] = useState('created_desc');
  const [status, setStatus] = useState('');
  const [page, setPage] = useState(0);
  const toast = useToast();

  // Debounce the search box so typing does not fire a request per keystroke.
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebounced(search);
      setPage(0);
    }, 350);
    return () => clearTimeout(timer);
  }, [search]);

  const { data, loading, error, reload } = useApi(
    () =>
      api.listPredictions({
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
        search: debounced || undefined,
        status: status || undefined,
        sort,
      }),
    [page, debounced, status, sort],
  );

  const remove = useCallback(
    async (id) => {
      try {
        await api.deletePrediction(id);
        toast.push('Prediction deleted.', { type: 'success' });
        reload();
      } catch (err) {
        toast.push(err.message, { type: 'error' });
      }
    },
    [reload, toast],
  );

  const clearAll = useCallback(async () => {
    if (!window.confirm('Delete every stored prediction and its images? This cannot be undone.')) {
      return;
    }
    try {
      const result = await api.clearPredictions();
      toast.push(result.message, { type: 'success' });
      setPage(0);
      reload();
    } catch (err) {
      toast.push(err.message, { type: 'error' });
    }
  }, [reload, toast]);

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 0;

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-10 sm:px-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Prediction history</h1>
          <p className="mt-1.5 text-secondary">
            {data ? `${data.total} stored analysis${data.total === 1 ? '' : 'es'}` : 'Loading…'}
          </p>
        </div>
        {data?.total > 0 && (
          <Button variant="danger" size="sm" icon={Trash2} onClick={clearAll}>
            Clear history
          </Button>
        )}
      </header>

      <Card>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="relative lg:col-span-2">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search by disease, plant or condition…"
              className="w-full rounded-lg border border-[var(--border-strong)] bg-[var(--surface-raised)] py-2.5 pl-9 pr-3 text-sm"
            />
          </div>
          <select
            value={status}
            onChange={(event) => {
              setStatus(event.target.value);
              setPage(0);
            }}
            className="rounded-lg border border-[var(--border-strong)] bg-[var(--surface-raised)] px-3 py-2.5 text-sm"
          >
            {STATUSES.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <select
            value={sort}
            onChange={(event) => setSort(event.target.value)}
            className="rounded-lg border border-[var(--border-strong)] bg-[var(--surface-raised)] px-3 py-2.5 text-sm"
          >
            {SORTS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </Card>

      {loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <Card key={index} className="p-0 overflow-hidden">
              <Skeleton className="aspect-[4/3] w-full rounded-none" />
              <div className="p-4">
                <Skeleton className="mb-2 h-4 w-2/3" />
                <Skeleton className="h-3 w-1/3" />
              </div>
            </Card>
          ))}
        </div>
      ) : error ? (
        <ErrorState error={error} onRetry={reload} />
      ) : data.items.length === 0 ? (
        <Card>
          <EmptyState
            icon={debounced || status ? Filter : HistoryIcon}
            title={debounced || status ? 'No matching predictions' : 'No predictions yet'}
            description={
              debounced || status
                ? 'Try a different search term or clear the filters.'
                : 'Analysed leaves are stored here so you can revisit them later.'
            }
            action={
              !debounced && !status ? (
                <Link to="/">
                  <Button icon={ScanLine}>Analyse a leaf</Button>
                </Link>
              ) : (
                <Button
                  variant="secondary"
                  onClick={() => {
                    setSearch('');
                    setStatus('');
                  }}
                >
                  Clear filters
                </Button>
              )
            }
          />
        </Card>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {data.items.map((item) => {
              const style = CONFIDENCE_STYLES[item.confidence_level] || CONFIDENCE_STYLES.low;
              const badge = STATUS_BADGES[item.status];
              return (
                <Card key={item.id} className="group flex flex-col overflow-hidden p-0">
                  <Link to={`/history/${item.id}`} className="block">
                    <div className="aspect-[4/3] overflow-hidden bg-[var(--surface-sunken)]">
                      <img
                        src={imageUrl(`/api/images/${item.image_filename}`)}
                        alt={prettyClass(item.predicted_class)}
                        loading="lazy"
                        className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                      />
                    </div>
                  </Link>
                  <div className="flex flex-1 flex-col p-4">
                    {badge && (
                      <Chip className={clsx('mb-2 self-start', badge.className)}>
                        {badge.label}
                      </Chip>
                    )}
                    <div className="mb-2 flex items-start justify-between gap-2">
                      <Link to={`/history/${item.id}`} className="min-w-0 hover:text-leaf-600">
                        <p
                          className={clsx(
                            'truncate font-medium',
                            badge && 'text-muted line-through decoration-1',
                          )}
                        >
                          {item.predicted_condition}
                        </p>
                        <p className="truncate text-sm text-muted">{item.predicted_plant}</p>
                      </Link>
                      <Chip className={clsx('shrink-0', badge ? 'opacity-60' : '', style.chip)}>
                        {percent(item.confidence, 1)}
                      </Chip>
                    </div>
                    <div className="mt-auto flex items-center justify-between gap-2 pt-2 text-xs text-muted">
                      <span title={formatDate(item.created_at)}>{relativeTime(item.created_at)}</span>
                      <button
                        onClick={() => remove(item.id)}
                        className="rounded p-1 opacity-60 transition hover:bg-clay-100 hover:text-clay-700 hover:opacity-100 dark:hover:bg-clay-950"
                        aria-label={`Delete prediction ${item.id}`}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>

          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-3">
              <Button
                variant="secondary"
                size="sm"
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
              >
                Previous
              </Button>
              <span className="text-sm text-secondary">
                Page {page + 1} of {totalPages}
              </span>
              <Button
                variant="secondary"
                size="sm"
                disabled={page >= totalPages - 1}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */

export function HistoryDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const { data, loading, error, reload } = useApi(() => api.getPrediction(id), [id]);

  if (loading) {
    return (
      <div className="mx-auto max-w-5xl space-y-6 px-4 py-10 sm:px-6">
        <SkeletonCard lines={4} />
        <SkeletonCard lines={8} />
      </div>
    );
  }
  if (error) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-10 sm:px-6">
        <ErrorState error={error} onRetry={reload} title="Prediction not found" />
      </div>
    );
  }

  const isRejected = data.status === 'not_a_plant' || data.status === 'poor_quality';
  const style = isRejected
    ? {
        chip:
          'border-clay-300 bg-clay-50 text-clay-900 '
          + 'dark:border-clay-800 dark:bg-clay-950/60 dark:text-clay-100',
        label: 'Photo Rejected',
      }
    : CONFIDENCE_STYLES[data.confidence_level] || CONFIDENCE_STYLES.low;
  const files = data.explanation_files || {};
  const methods = Object.keys(files).filter((key) => key !== 'original');

  const remove = async () => {
    if (!window.confirm('Delete this prediction?')) return;
    try {
      await api.deletePrediction(id);
      toast.push('Prediction deleted.', { type: 'success' });
      navigate('/history');
    } catch (err) {
      toast.push(err.message, { type: 'error' });
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-10 sm:px-6">
      <div className="flex items-center justify-between gap-3">
        <Link to="/history" className="inline-flex items-center gap-2 text-sm text-secondary hover:text-leaf-600">
          <ArrowLeft size={15} />
          Back to history
        </Link>
        <Button variant="danger" size="sm" icon={Trash2} onClick={remove}>
          Delete
        </Button>
      </div>

      <Card className="overflow-hidden p-0">
        <div className="grid lg:grid-cols-[minmax(0,20rem)_minmax(0,1fr)]">
          <div className="bg-black/5 p-4 dark:bg-black/30">
            <img
              src={imageUrl(files.original || `/api/images/${data.image_filename}`)}
              alt={prettyClass(data.predicted_class)}
              className="w-full rounded-xl object-contain"
            />
          </div>
          <div className="p-5 sm:p-6">
            <div className="flex flex-wrap items-center gap-2">
              <Chip className={style.chip}>{style.label}</Chip>
              {isRejected && (
                <Chip className="border-red-300 bg-red-100 text-red-800 dark:border-red-800 dark:bg-red-900/60 dark:text-red-200">
                  {data.status === 'not_a_plant' ? 'Not a plant leaf' : 'Poor image quality'}
                </Chip>
              )}
            </div>
            <p className="mt-3 text-sm uppercase tracking-wide text-muted">
              {isRejected ? 'Subject' : data.predicted_plant}
            </p>
            <h1 className="mt-1 text-2xl font-semibold">
              {isRejected ? 'No Diagnosis Provided' : data.predicted_condition}
            </h1>
            {!isRejected ? (
              <>
                <p className="mt-1 font-mono text-xs text-muted">{data.predicted_class}</p>
                <p className="mt-4 text-3xl font-semibold tabular-nums">{percent(data.confidence, 2)}</p>
              </>
            ) : (
              <p className="mt-2 text-sm text-secondary leading-relaxed">
                This image was rejected by our input verification gate. Only genuine leaves of supported agricultural crops are diagnosed.
              </p>
            )}

            <dl className="mt-5 grid grid-cols-2 gap-3 text-sm">
              <Detail label="Analysed" value={formatDate(data.created_at)} />
              <Detail label="Model" value={data.model_label || data.model_name} />
              <Detail label="Inference" value={milliseconds(data.inference_ms)} />
              <Detail label="Total" value={milliseconds(data.total_ms)} />
              <Detail label="Image quality" value={`${percent(data.image_quality_score, 0)} (${data.image_quality_band})`} />
              <Detail label="Status" value={data.status} />
            </dl>

            {data.status_message && (
              <p className="mt-4 rounded-lg bg-[var(--surface-sunken)] p-3 text-sm text-secondary">
                {data.status_message}
              </p>
            )}
          </div>
        </div>
      </Card>

      {data.top_predictions?.length > 0 && (
        <Card>
          <CardHeader title="Top predictions" icon={Clock} />
          <ul className="space-y-2">
            {data.top_predictions.map((item, index) => (
              <li key={item.class_name} className="flex items-center justify-between gap-3 text-sm">
                <span className={clsx('truncate', index === 0 && 'font-medium')}>
                  <span className="mr-2 text-muted">{index + 1}.</span>
                  {item.display_name || prettyClass(item.class_name)}
                </span>
                <span className="tabular-nums">{percent(item.probability, 2)}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {methods.length > 0 && (
        <Card>
          <CardHeader title="Saved explanations" subtitle="Heat maps generated at analysis time" />
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {methods.map((method) => (
              <figure key={method}>
                <img
                  src={imageUrl(files[method])}
                  alt={method}
                  loading="lazy"
                  className="w-full rounded-lg border border-[var(--border-subtle)]"
                />
                <figcaption className="mt-1.5 text-xs text-muted">
                  {method.replace(/_/g, ' ')}
                </figcaption>
              </figure>
            ))}
          </div>
        </Card>
      )}

      <DiseaseInfoPanel info={data.disease_info} />
    </div>
  );
}

function Detail({ label, value }) {
  return (
    <div>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="mt-0.5 truncate font-medium">{value}</dd>
    </div>
  );
}
