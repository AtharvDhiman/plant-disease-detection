import {
  Activity,
  BarChart3,
  Clock,
  Gauge,
  Layers,
  Leaf,
  PieChart as PieIcon,
  ScanLine,
  TrendingUp,
} from 'lucide-react';
import { Link } from 'react-router-dom';

import {
  DonutChart,
  HorizontalBarChart,
  SimpleBarChart,
  TimelineChart,
} from '../components/charts';
import {
  Card,
  CardHeader,
  Chip,
  EmptyState,
  ErrorState,
  SkeletonCard,
  SkeletonStats,
  StatCard,
  Button,
} from '../components/ui';
import { useApi } from '../hooks/useApi';
import { api, imageUrl } from '../services/api';
import {
  CONFIDENCE_STYLES,
  milliseconds,
  percent,
  prettyClass,
  relativeTime,
} from '../utils/format';

export default function Dashboard() {
  const { data, loading, error, reload } = useApi(() => api.dashboard(30), []);
  // Benchmark rows and per-class scores come from the training artefacts, not
  // from the prediction history, so they render even on a fresh install.
  const { data: metrics } = useApi(() => api.metrics('benchmark').catch(() => null), []);
  const { data: perClass } = useApi(() => api.perClass().catch(() => null), []);

  if (loading) {
    return (
      <div className="mx-auto max-w-7xl space-y-6 px-4 py-10 sm:px-6">
        <SkeletonStats />
        <div className="grid gap-6 lg:grid-cols-2">
          <SkeletonCard lines={6} />
          <SkeletonCard lines={6} />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-10 sm:px-6">
        <ErrorState error={error} onRetry={reload} title="Could not load the dashboard" />
      </div>
    );
  }

  const totals = data.totals;
  const isEmpty = totals.predictions === 0;

  const diseaseData = data.disease_distribution
    .slice(0, 8)
    .map((item) => ({ name: prettyClass(item.class_name), value: item.count }));

  const plantData = data.plant_distribution
    .slice(0, 10)
    .map((item) => ({ name: item.plant, count: item.count }));

  const confidenceData = data.confidence_histogram.map((bin) => ({
    name: bin.range,
    count: bin.count,
  }));

  const modelRows = (metrics?.rows || [])
    .filter((row) => row.accuracy != null)
    .slice(0, 8)
    .map((row) => ({
      ...row,
      label: row.label.length > 26 ? `${row.label.slice(0, 25)}…` : row.label,
    }));

  // Ascending F1: the classes the model finds hardest are the useful ones to see.
  const classRows = (perClass?.classes || [])
    .slice(0, 8)
    .map((row) => ({ label: prettyClass(row.class_name), value: row.f1 }));

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-10 sm:px-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Dashboard</h1>
          <p className="mt-1.5 text-secondary">
            Live statistics over every analysis this installation has performed.
          </p>
        </div>
        <Button variant="secondary" size="sm" onClick={reload}>
          Refresh
        </Button>
      </header>

      {/* --------------------------------------------------------- headline */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Total predictions"
          value={totals.predictions.toLocaleString()}
          sub={`${totals.distinct_classes} distinct classes`}
          icon={ScanLine}
        />
        <StatCard
          label="Average confidence"
          value={percent(totals.average_confidence, 1)}
          sub={`${totals.healthy} healthy · ${totals.diseased} diseased`}
          icon={Gauge}
          tone={totals.average_confidence >= 0.85 ? 'leaf' : 'amber'}
        />
        <StatCard
          label="Most detected"
          value={data.most_detected ? prettyClass(data.most_detected.class_name) : '—'}
          sub={data.most_detected ? `${data.most_detected.count} detections` : 'No data yet'}
          icon={Leaf}
          tone="clay"
        />
        <StatCard
          label="Average response"
          value={milliseconds(totals.average_total_ms)}
          sub={`${milliseconds(totals.average_inference_ms)} model inference`}
          icon={Clock}
          tone="neutral"
        />
      </div>

      {/* ------------------------------------------------------ model panel */}
      <Card>
        <CardHeader
          title="Serving model"
          subtitle="Selected from the benchmark by measured performance"
          icon={Layers}
          action={
            <Link to="/benchmark">
              <Button variant="secondary" size="sm">
                Full benchmark
              </Button>
            </Link>
          }
        />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Model" value={data.model.label || '—'} />
          <Field label="Test accuracy" value={percent(data.model.accuracy, 2)} />
          <Field label="Macro F1" value={data.model.f1_macro?.toFixed(4) ?? '—'} />
          <Field label="Inference latency" value={milliseconds(data.model.inference_ms)} />
        </div>
        {data.model.best_benchmark_model && data.model.best_benchmark_model !== data.model.label && (
          <p className="mt-4 rounded-lg bg-[var(--surface-sunken)] p-3 text-sm text-secondary">
            The highest-accuracy model in the benchmark is{' '}
            <strong>{data.model.best_benchmark_model}</strong> at{' '}
            {percent(data.model.best_benchmark_accuracy, 2)}. The serving model is chosen by the
            criterion recorded in the export manifest, which may weigh size and latency as well as
            accuracy.
          </p>
        )}
      </Card>

      {isEmpty ? (
        <Card>
          <EmptyState
            icon={ScanLine}
            title="No analyses yet"
            description="Upload a leaf image to populate the dashboard. Every prediction you run is recorded here."
            action={
              <Link to="/">
                <Button icon={ScanLine}>Analyse a leaf</Button>
              </Link>
            }
          />
        </Card>
      ) : (
        <>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader
                title="Detected conditions"
                subtitle="Distribution across all analyses"
                icon={PieIcon}
              />
              <DonutChart data={diseaseData} />
            </Card>

            <Card>
              <CardHeader
                title="Confidence distribution"
                subtitle="How certain the model has been"
                icon={BarChart3}
              />
              <SimpleBarChart data={confidenceData} dataKey="count" nameKey="name" />
              <div className="mt-3 flex flex-wrap gap-2">
                {data.confidence_levels.map((level) => (
                  <Chip key={level.level} className={CONFIDENCE_STYLES[level.level]?.chip}>
                    {level.level}: {level.count}
                  </Chip>
                ))}
              </div>
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader title="Plants analysed" icon={Leaf} />
              <SimpleBarChart data={plantData} dataKey="count" nameKey="name" />
            </Card>

            <Card>
              <CardHeader
                title={`Activity (last ${data.window_days} days)`}
                subtitle="Predictions per day"
                icon={TrendingUp}
              />
              {data.timeline.length > 0 ? (
                <TimelineChart data={data.timeline} />
              ) : (
                <EmptyState icon={Activity} title="No activity in this window" />
              )}
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader
                title="Model comparison"
                subtitle="Test accuracy across the benchmark"
                icon={Layers}
                action={
                  <Link to="/benchmark">
                    <Button variant="secondary" size="sm">
                      Details
                    </Button>
                  </Link>
                }
              />
              {modelRows.length ? (
                <HorizontalBarChart
                  data={modelRows}
                  dataKey="accuracy"
                  height={Math.max(220, modelRows.length * 34)}
                  formatter={(v) => percent(v, 2)}
                  domain={[
                    Math.max(0, Math.min(...modelRows.map((r) => r.accuracy)) - 0.04),
                    1,
                  ]}
                />
              ) : (
                <EmptyState icon={Layers} title="No benchmark results yet" />
              )}
            </Card>

            <Card>
              <CardHeader
                title="Class-wise performance"
                subtitle={
                  perClass ? `Hardest classes for ${perClass.label}` : 'Per-class F1'
                }
                icon={Activity}
                action={
                  <Link to="/research">
                    <Button variant="secondary" size="sm">
                      All classes
                    </Button>
                  </Link>
                }
              />
              {classRows.length ? (
                <HorizontalBarChart
                  data={classRows}
                  dataKey="value"
                  height={Math.max(220, classRows.length * 34)}
                  formatter={(v) => v.toFixed(4)}
                  highlightKey="__none"
                  domain={[0, 1]}
                />
              ) : (
                <EmptyState icon={Activity} title="Per-class metrics not available" />
              )}
            </Card>
          </div>

          <Card>
            <CardHeader
              title="Recent analyses"
              icon={Clock}
              action={
                <Link to="/history">
                  <Button variant="secondary" size="sm">
                    View all
                  </Button>
                </Link>
              }
            />
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {data.recent.map((item) => {
                const style = CONFIDENCE_STYLES[item.confidence_level] || CONFIDENCE_STYLES.low;
                return (
                  <Link
                    key={item.id}
                    to={`/history/${item.id}`}
                    className="group overflow-hidden rounded-xl border border-[var(--border-subtle)] transition-shadow hover:shadow-md"
                  >
                    <div className="aspect-[4/3] overflow-hidden bg-[var(--surface-sunken)]">
                      <img
                        src={imageUrl(item.image_url)}
                        alt={item.display_name}
                        loading="lazy"
                        className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                      />
                    </div>
                    <div className="p-3">
                      <p className="truncate text-sm font-medium">{item.display_name}</p>
                      <div className="mt-1.5 flex items-center justify-between gap-2">
                        <span className="text-xs text-muted">{relativeTime(item.created_at)}</span>
                        <span className={`inline-flex h-1.5 w-1.5 rounded-full ${style.dot}`} />
                      </div>
                      <p className="mt-1 text-xs tabular-nums text-secondary">
                        {percent(item.confidence, 1)}
                      </p>
                    </div>
                  </Link>
                );
              })}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

function Field({ label, value }) {
  return (
    <div className="rounded-lg border border-[var(--border-subtle)] p-3">
      <p className="text-xs text-muted">{label}</p>
      <p className="mt-1 truncate font-semibold tabular-nums">{value}</p>
    </div>
  );
}
