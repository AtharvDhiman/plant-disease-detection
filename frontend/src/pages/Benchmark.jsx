import clsx from 'clsx';
import { Activity, ArrowUpDown, Award, Cpu, Gauge, Star, Timer, Zap } from 'lucide-react';
import { useMemo, useState } from 'react';

import { HorizontalBarChart, TradeoffScatter } from '../components/charts';
import {
  Button,
  Card,
  CardHeader,
  Chip,
  EmptyState,
  ErrorState,
  InfoTooltip,
  SkeletonCard,
  StatCard,
  Tabs,
} from '../components/ui';
import { useApi } from '../hooks/useApi';
import { api } from '../services/api';
import { decimal, megabytes, milliseconds, parameters, percent } from '../utils/format';

const COLUMNS = [
  { key: 'label', label: 'Model', sortable: false, align: 'left' },
  { key: 'accuracy', label: 'Accuracy', format: (v) => percent(v, 2) },
  { key: 'precision', label: 'Precision', format: (v) => decimal(v, 4) },
  { key: 'recall', label: 'Recall', format: (v) => decimal(v, 4) },
  { key: 'f1_macro', label: 'Macro F1', format: (v) => decimal(v, 4) },
  { key: 'top3_accuracy', label: 'Top-3', format: (v) => percent(v, 2) },
  { key: 'parameters', label: 'Parameters', format: parameters },
  { key: 'model_size_mb', label: 'Size', format: megabytes },
  { key: 'inference_ms', label: 'Latency', format: milliseconds },
  { key: 'ece', label: 'ECE', format: (v) => decimal(v, 4), tooltip: 'Expected calibration error: the average gap between stated confidence and observed accuracy. Lower is better.' },
];

const GROUPS = [
  { id: 'all', label: 'All models' },
  { id: 'benchmark', label: 'Deep learning' },
  { id: 'classical', label: 'Classical ML' },
  { id: 'ablation', label: 'Ablation arms' },
  { id: 'placement', label: 'CBAM placement' },
];

export default function Benchmark() {
  const { data, loading, error, reload } = useApi(() => api.metrics(), []);
  const { data: models } = useApi(() => api.models().catch(() => null), []);
  const [group, setGroup] = useState('all');
  const [sortKey, setSortKey] = useState('accuracy');
  const [sortDesc, setSortDesc] = useState(true);

  const rows = useMemo(() => {
    if (!data?.rows) return [];
    const filtered = group === 'all' ? data.rows : data.rows.filter((row) => row.group === group);
    return [...filtered].sort((a, b) => {
      const left = a[sortKey] ?? -Infinity;
      const right = b[sortKey] ?? -Infinity;
      return sortDesc ? right - left : left - right;
    });
  }, [data, group, sortKey, sortDesc]);

  if (loading) {
    return (
      <div className="mx-auto max-w-7xl space-y-6 px-4 py-10 sm:px-6">
        <SkeletonCard lines={2} />
        <SkeletonCard lines={10} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-10 sm:px-6">
        <ErrorState error={error} onRetry={reload} title="Could not load the benchmark" />
      </div>
    );
  }

  const all = data.rows || [];
  const best = all.length ? all.reduce((a, b) => ((a.accuracy || 0) > (b.accuracy || 0) ? a : b)) : null;
  const bestF1 = all.length ? all.reduce((a, b) => ((a.f1_macro || 0) > (b.f1_macro || 0) ? a : b)) : null;
  const withLatency = all.filter((r) => r.inference_ms);
  const fastest = withLatency.length
    ? withLatency.reduce((a, b) => (a.inference_ms < b.inference_ms ? a : b))
    : null;
  const withSize = all.filter((r) => r.model_size_mb);
  const smallest = withSize.length
    ? withSize.reduce((a, b) => (a.model_size_mb < b.model_size_mb ? a : b))
    : null;

  const chartRows = rows
    .filter((r) => r.accuracy != null)
    .map((r) => ({ ...r, label: r.label.length > 30 ? `${r.label.slice(0, 29)}…` : r.label }));

  // Math.min of an empty array is Infinity, which would produce an unusable
  // axis domain. Selecting a group with no trained models is a normal state
  // (the ablation suite may not have run yet), so handle it rather than guard.
  const axisFloor = (key) =>
    chartRows.length === 0
      ? 0
      : Math.max(0, Math.min(...chartRows.map((r) => r[key] ?? 1)) - 0.05);

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-10 sm:px-6">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight">Model benchmark</h1>
        <p className="mt-2 max-w-3xl text-secondary">
          Every architecture below was trained and evaluated under one identical protocol — same
          data subset, same image size, same optimiser, same epoch budget, same seed. Only the
          architecture changes, which is what makes this a comparison rather than a leaderboard of
          unrelated runs.
        </p>
      </header>

      {all.length === 0 ? (
        <Card>
          <EmptyState
            icon={Activity}
            title="No benchmark results yet"
            description="Run python training/run_experiments.py --suite research to populate this page."
          />
        </Card>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label="Best accuracy"
              value={percent(best?.accuracy, 2)}
              sub={best?.label}
              icon={Award}
            />
            <StatCard
              label="Best macro F1"
              value={decimal(bestF1?.f1_macro, 4)}
              sub={bestF1?.label}
              icon={Star}
              tone="clay"
            />
            <StatCard
              label="Fastest"
              value={milliseconds(fastest?.inference_ms)}
              sub={fastest?.label}
              icon={Zap}
              tone="amber"
            />
            <StatCard
              label="Smallest"
              value={megabytes(smallest?.model_size_mb)}
              sub={smallest?.label}
              icon={Cpu}
              tone="neutral"
            />
          </div>

          {models?.selection && (
            <Card className="border-leaf-300 dark:border-leaf-800">
              <CardHeader
                title={`Serving: ${models.production?.label}`}
                subtitle={models.selection.criterion_description}
                icon={Gauge}
              />
              <p className="text-sm text-secondary">{models.selection.note}</p>
              {models.best_by && (
                <div className="mt-4 flex flex-wrap gap-2">
                  {Object.entries(models.best_by).map(([criterion, name]) => (
                    <Chip
                      key={criterion}
                      className="border-[var(--border-strong)] bg-[var(--surface-sunken)]"
                    >
                      <span className="text-muted">best {criterion}:</span> {name}
                    </Chip>
                  ))}
                </div>
              )}
            </Card>
          )}

          <div className="flex flex-wrap items-center justify-between gap-3">
            <Tabs tabs={GROUPS} active={group} onChange={setGroup} />
            <p className="text-sm text-muted">
              {rows.length} model{rows.length === 1 ? '' : 's'}
            </p>
          </div>

          {/* -------------------------------------------------------- table */}
          <Card className="p-0 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[var(--border-subtle)] bg-[var(--surface-sunken)]">
                    {COLUMNS.map((column) => (
                      <th
                        key={column.key}
                        className={clsx(
                          'whitespace-nowrap px-4 py-3 font-medium text-[var(--text-secondary)]',
                          column.align === 'left' ? 'text-left' : 'text-right',
                        )}
                      >
                        {column.sortable === false ? (
                          column.label
                        ) : (
                          <button
                            onClick={() => {
                              if (sortKey === column.key) setSortDesc((v) => !v);
                              else {
                                setSortKey(column.key);
                                setSortDesc(true);
                              }
                            }}
                            className={clsx(
                              'inline-flex items-center gap-1 hover:text-[var(--text-primary)]',
                              sortKey === column.key && 'text-[var(--text-primary)]',
                            )}
                          >
                            {column.label}
                            {column.tooltip && <InfoTooltip text={column.tooltip} />}
                            <ArrowUpDown size={12} className="opacity-50" />
                          </button>
                        )}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr
                      key={row.experiment_id}
                      className={clsx(
                        'border-b border-[var(--border-subtle)] last:border-0 transition-colors hover:bg-[var(--surface-sunken)]',
                        row.proposed && 'bg-clay-50/50 dark:bg-clay-950/25',
                      )}
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {row.proposed && <Star size={13} className="shrink-0 text-clay-600" fill="currentColor" />}
                          <div className="min-w-0">
                            <p className={clsx('truncate', row.proposed && 'font-semibold')}>{row.label}</p>
                            <p className="text-xs text-muted">
                              {row.family === 'classical' ? 'hand-crafted features' : row.group}
                              {row.protocol === 'production' && ' · full data'}
                            </p>
                          </div>
                        </div>
                      </td>
                      {COLUMNS.slice(1).map((column) => (
                        <td
                          key={column.key}
                          className="whitespace-nowrap px-4 py-3 text-right tabular-nums"
                        >
                          {row[column.key] == null ? (
                            <span className="text-muted">—</span>
                          ) : (
                            column.format(row[column.key])
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <p className="text-xs text-muted">
            <Star size={11} className="mb-0.5 mr-1 inline text-clay-600" fill="currentColor" />
            marks the proposed CBAM architectures. {data.protocol_note}
          </p>

          {/* ------------------------------------------------------- charts */}
          {chartRows.length === 0 ? (
            <Card>
              <EmptyState
                icon={Activity}
                title="No models in this group yet"
                description="Run the corresponding training suite to populate it."
              />
            </Card>
          ) : (
          <>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader title="Accuracy" subtitle="Held-out test split" icon={Award} />
              <HorizontalBarChart
                data={chartRows}
                dataKey="accuracy"
                height={Math.max(280, chartRows.length * 34)}
                formatter={(v) => percent(v, 2)}
                domain={[axisFloor('accuracy'), 1]}
              />
            </Card>

            <Card>
              <CardHeader title="Macro F1" subtitle="Every class weighted equally" icon={Star} />
              <HorizontalBarChart
                data={chartRows.filter((r) => r.f1_macro != null)}
                dataKey="f1_macro"
                height={Math.max(280, chartRows.length * 34)}
                formatter={(v) => decimal(v, 4)}
                domain={[axisFloor('f1_macro'), 1]}
              />
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader
                title="Model size vs accuracy"
                subtitle="Points nearer the top-left are better value"
                icon={Cpu}
              />
              <TradeoffScatter
                data={chartRows.filter((r) => r.model_size_mb)}
                xKey="model_size_mb"
                yKey="accuracy"
                xLabel="Size (MB)"
                yLabel="Accuracy"
                logX
                formatter={{ x: (v) => megabytes(v), y: (v) => percent(v, 2) }}
              />
            </Card>

            <Card>
              <CardHeader
                title="Latency vs accuracy"
                subtitle="Single-image forward pass on the training device"
                icon={Timer}
              />
              <TradeoffScatter
                data={chartRows.filter((r) => r.inference_ms)}
                xKey="inference_ms"
                yKey="accuracy"
                xLabel="ms / image"
                yLabel="Accuracy"
                formatter={{ x: (v) => milliseconds(v), y: (v) => percent(v, 2) }}
              />
            </Card>
          </div>

          </>
          )}

          <div className="flex justify-center">
            <Button variant="secondary" onClick={reload}>
              Refresh results
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
