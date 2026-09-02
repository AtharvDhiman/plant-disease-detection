import clsx from 'clsx';
import {
  Beaker,
  Binary,
  Database,
  FlaskConical,
  Gauge,
  GitCompare,
  Grid3x3,
  Images,
  LineChart,
  Microscope,
  HardDrive,
  ShieldAlert,
  Sparkles,
  Timer,
  Cpu,
  Zap,
} from 'lucide-react';
import { useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';

import {
  GroupedBarChart,
  HorizontalBarChart,
  ReliabilityChart,
  SimpleBarChart,
  TrainingCurveChart,
} from '../components/charts';
import {
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
import { api, figureUrl } from '../services/api';
import { decimal, duration, number, percent, prettyClass } from '../utils/format';

const TABS = [
  { id: 'dataset', label: 'Dataset' },
  { id: 'ablation', label: 'Ablation study' },
  { id: 'training', label: 'Training curves' },
  { id: 'calibration', label: 'Calibration & OOD' },
  { id: 'perclass', label: 'Per-class' },
  { id: 'performance', label: 'Performance' },
  { id: 'figures', label: 'Figures' },
];

// How confident the ablation verdict is allowed to look. The tone comes from the
// measured evidence strength in the ledger rather than a threshold hardcoded
// here, so the UI cannot disagree with the report about what the data supports.
const EVIDENCE_LABEL = {
  supported: 'Supported by the measurement',
  suggestive: 'Suggestive, not established',
  'within noise': 'Within run-to-run noise',
  'no improvement': 'No improvement measured',
  unknown: 'Strength not assessed',
};

const EVIDENCE_BORDER = {
  supported: 'border-leaf-300 dark:border-leaf-800',
  suggestive: 'border-amber-300 dark:border-amber-800',
  'within noise': 'border-amber-300 dark:border-amber-800',
  'no improvement': 'border-rose-300 dark:border-rose-900',
  unknown: 'border-[var(--border)]',
};

const EVIDENCE_BADGE = {
  supported: 'bg-leaf-100 text-leaf-800 dark:bg-leaf-950 dark:text-leaf-200',
  suggestive: 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200',
  'within noise': 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200',
  'no improvement': 'bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-200',
  unknown: 'bg-[var(--surface-sunken)] text-secondary',
};

export default function Research() {
  // The tab lives in the URL, not in component state. A research dashboard
  // whose ablation study cannot be linked to is hard to share and loses its
  // place on a back-navigation or reload; it also cannot be screenshotted by
  // the documentation script, which only knows how to visit a path.
  const [searchParams, setSearchParams] = useSearchParams();
  const requested = searchParams.get('tab');
  const tab = TABS.some((t) => t.id === requested) ? requested : 'dataset';
  const setTab = (next) => {
    // replace, not push: flipping through tabs should not fill the history
    // stack with entries the back button has to walk through.
    setSearchParams(next === 'dataset' ? {} : { tab: next }, { replace: true });
  };

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-10 sm:px-6">
      <header>
        <div className="flex items-center gap-2.5">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-leaf-100 text-leaf-700 dark:bg-leaf-900/60 dark:text-leaf-300">
            <FlaskConical size={20} />
          </span>
          <h1 className="text-3xl font-semibold tracking-tight">Research dashboard</h1>
        </div>
        <p className="mt-3 max-w-3xl text-secondary">
          Everything on this page is read from the experiment artefacts produced by the training
          pipeline. No figure here is hand-written — if an experiment has not been run, the section
          says so rather than showing a placeholder.
        </p>
      </header>

      <Tabs tabs={TABS} active={tab} onChange={setTab} />

      {tab === 'dataset' && <DatasetSection />}
      {tab === 'ablation' && <AblationSection />}
      {tab === 'training' && <TrainingSection />}
      {tab === 'calibration' && <CalibrationSection />}
      {tab === 'perclass' && <PerClassSection />}
      {tab === 'performance' && <PerformanceSection />}
      {tab === 'figures' && <FiguresSection />}
    </div>
  );
}

/* ------------------------------------------------------------------ dataset */

function DatasetSection() {
  const { data, loading, error, reload } = useApi(() => api.datasetStats(), []);

  if (loading) return <SkeletonCard lines={8} />;
  if (error) return <ErrorState error={error} onRetry={reload} title="Dataset report unavailable" />;

  const classCounts = Object.entries(data.class_counts || {})
    .map(([name, count]) => ({ label: prettyClass(name), value: count }))
    .sort((a, b) => b.value - a.value);

  const leak = data.duplicates?.cross_split_leakage;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Classes" value={data.num_classes} sub={`${data.num_plants} plant species`} icon={Grid3x3} />
        <StatCard label="Total images" value={number(data.total_images)} sub="across all official splits" icon={Images} />
        <StatCard
          label="Imbalance ratio"
          value={decimal(data.imbalance.imbalance_ratio, 2)}
          sub={`min ${number(data.imbalance.min_count)} · max ${number(data.imbalance.max_count)}`}
          icon={Binary}
          tone={data.imbalance.imbalance_ratio > 2 ? 'amber' : 'leaf'}
        />
        <StatCard
          label="Healthy / diseased"
          value={`${data.healthy_classes} / ${data.diseased_classes}`}
          sub="class split"
          icon={Beaker}
          tone="neutral"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Split protocol" subtitle="How the data was divided" icon={Database} />
          <p className="text-sm leading-relaxed text-secondary">{data.protocol?.description}</p>
          <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
            {Object.entries(data.working_splits || {}).map(([name, count]) => (
              <div key={name} className="rounded-lg border border-[var(--border-subtle)] p-3">
                <p className="text-xs text-muted">{name}</p>
                <p className="mt-1 font-semibold tabular-nums">{number(count)}</p>
              </div>
            ))}
          </div>
          {data.protocol?.benchmark_subset && (
            <div className="mt-4 rounded-lg bg-[var(--surface-sunken)] p-3.5">
              <p className="mb-1.5 text-sm font-medium">Why a benchmark subset?</p>
              <p className="text-sm leading-relaxed text-secondary">
                {data.protocol.benchmark_subset.why}
              </p>
              <p className="mt-2 text-xs text-muted">
                {data.protocol.benchmark_subset.train_per_class} train /{' '}
                {data.protocol.benchmark_subset.val_per_class} val /{' '}
                {data.protocol.benchmark_subset.test_per_class} test images per class.
              </p>
            </div>
          )}
        </Card>

        <Card>
          <CardHeader
            title="Class imbalance handling"
            subtitle="Chosen from the measured distribution"
            icon={Binary}
          />
          <div className="flex items-center gap-2">
            <Chip className="border-leaf-300 bg-leaf-100 text-leaf-800 dark:border-leaf-800 dark:bg-leaf-900/60 dark:text-leaf-200">
              strategy: {data.imbalance_strategy || 'none'}
            </Chip>
            <Chip className="border-[var(--border-strong)] bg-[var(--surface-sunken)]">
              CV {decimal(data.imbalance.coefficient_of_variation, 3)}
            </Chip>
          </div>
          <p className="mt-3 text-sm leading-relaxed text-secondary">{data.imbalance_reason}</p>

          <div className="mt-5 border-t border-[var(--border-subtle)] pt-4">
            <p className="mb-2 flex items-center gap-1.5 text-sm font-medium">
              Data-leakage check
              <InfoTooltip text="Perceptual hashes (pHash) of a stratified sample from each split are compared. A collision across train and test would mean the same image appears in both, invalidating the reported accuracy." />
            </p>
            {leak ? (
              <div
                className={clsx(
                  'rounded-lg border p-3 text-sm',
                  leak.colliding_hashes === 0
                    ? 'border-leaf-300 bg-leaf-50 text-leaf-900 dark:border-leaf-800 dark:bg-leaf-950/60 dark:text-leaf-100'
                    : 'border-clay-300 bg-clay-50 text-clay-900 dark:border-clay-800 dark:bg-clay-950/60 dark:text-clay-100',
                )}
              >
                <p className="font-medium">
                  {leak.colliding_hashes === 0
                    ? 'No train/test overlap found in the sample'
                    : `${leak.colliding_hashes} colliding hashes found`}
                </p>
                <p className="mt-1 opacity-90">
                  {percent(leak.leak_rate_in_sample, 2)} of sampled test images collide with a
                  training image. This is a sampled estimate, not an exhaustive all-pairs check.
                </p>
              </div>
            ) : (
              <p className="text-sm text-muted">Leakage analysis not available.</p>
            )}
            <p className="mt-3 text-xs text-muted">
              Corrupt images in sample: {percent(data.corrupt_rate_in_sample, 2)} · Image
              dimensions: {(data.image_size || []).map((s) => `${s[0]}×${s[1]}`).join(', ')}
            </p>
          </div>
        </Card>
      </div>

      <Card>
        <CardHeader title="Images per class" subtitle="Training split" icon={Grid3x3} />
        <HorizontalBarChart
          data={classCounts}
          dataKey="value"
          height={Math.max(400, classCounts.length * 22)}
          formatter={(v) => number(v)}
          highlightKey="__none"
        />
      </Card>
    </div>
  );
}

/* ----------------------------------------------------------------- ablation */

function AblationSection() {
  const { data, loading, error, reload } = useApi(() => api.ablation(), []);

  if (loading) return <SkeletonCard lines={8} />;
  if (error) {
    return (
      <ErrorState
        error={error}
        onRetry={reload}
        title="Ablation results not available yet"
      />
    );
  }

  const verdict = data.summary?.verdict;
  const placement = data.summary?.placement;
  const arms = data.summary?.arms || [];
  const chartData = arms.map((arm) => ({
    label: `${arm.arm}: ${arm.label.replace('CNN + ', '').replace('CNN (control)', 'None')}`,
    accuracy: arm.accuracy,
    f1_macro: arm.f1_macro,
    recall_macro: arm.recall_macro,
  }));

  return (
    <div className="space-y-6">
      <Card className="border-leaf-300 dark:border-leaf-800">
        <CardHeader title="Research question" icon={Microscope} />
        <p className="text-lg leading-relaxed">{data.research_question}</p>
      </Card>

      {verdict ? (
        <Card className={EVIDENCE_BORDER[verdict.evidence_strength] ?? EVIDENCE_BORDER.unknown}>
          <CardHeader title="Answer from the experiments" icon={Sparkles} />
          {verdict.evidence_strength && (
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <span
                className={clsx(
                  'rounded-full px-2.5 py-1 text-xs font-medium',
                  EVIDENCE_BADGE[verdict.evidence_strength] ?? EVIDENCE_BADGE.unknown,
                )}
              >
                {EVIDENCE_LABEL[verdict.evidence_strength] ?? verdict.evidence_strength}
              </span>
              {verdict.margin_over_threshold != null && (
                <span className="text-xs text-secondary">
                  {verdict.margin_over_threshold.toFixed(1)}x the measured{' '}
                  {(verdict.significance_threshold_f1 * 100).toFixed(2)}-point noise floor
                </span>
              )}
            </div>
          )}
          <p className="text-base leading-relaxed">{verdict.statement}</p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Delta label="Macro F1 vs control" value={verdict.cbam_delta_f1_macro} />
            <Delta label="Accuracy vs control" value={verdict.cbam_delta_accuracy} />
            <Delta
              label="Inference overhead"
              value={verdict.cbam_inference_overhead_pct / 100}
              invert
              suffix="%"
              raw={verdict.cbam_inference_overhead_pct}
            />
            <Delta
              label="Parameter overhead"
              value={verdict.cbam_parameter_overhead_pct / 100}
              invert
              suffix="%"
              raw={verdict.cbam_parameter_overhead_pct}
            />
          </div>
          <p className="mt-4 rounded-lg bg-[var(--surface-sunken)] p-3 text-sm leading-relaxed text-secondary">
            <strong className="font-medium">Caveat: </strong>
            {verdict.caveat}
          </p>
        </Card>
      ) : (
        <Card>
          <EmptyState
            icon={Microscope}
            title="Ablation not complete"
            description="Run python training/run_experiments.py --suite ablation to compute the five arms."
          />
        </Card>
      )}

      {chartData.length > 0 && (
        <Card>
          <CardHeader
            title="Attention ablation"
            subtitle="Identical CNN backbone, identical training budget — only the attention block differs"
            icon={GitCompare}
          />
          <GroupedBarChart
            data={chartData}
            series={[
              { key: 'accuracy', label: 'Accuracy' },
              { key: 'f1_macro', label: 'Macro F1' },
              { key: 'recall_macro', label: 'Macro recall' },
            ]}
            height={380}
            formatter={(v) => decimal(v, 4)}
          />
        </Card>
      )}

      {arms.length > 0 && (
        <Card className="p-0 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--border-subtle)] bg-[var(--surface-sunken)] text-left">
                  <th className="px-4 py-3 font-medium">Arm</th>
                  <th className="px-4 py-3 font-medium">Configuration</th>
                  <th className="px-4 py-3 text-right font-medium">Accuracy</th>
                  <th className="px-4 py-3 text-right font-medium">Macro F1</th>
                  <th className="px-4 py-3 text-right font-medium">Δ F1</th>
                  <th className="px-4 py-3 text-right font-medium">Params</th>
                  <th className="px-4 py-3 text-right font-medium">ms/img</th>
                </tr>
              </thead>
              <tbody>
                {arms.map((arm) => (
                  <tr
                    key={arm.model_name}
                    className={clsx(
                      'border-b border-[var(--border-subtle)] last:border-0',
                      arm.model_name === 'abl_cnn_cbam' && 'bg-clay-50/50 dark:bg-clay-950/25',
                    )}
                  >
                    <td className="px-4 py-3 font-mono text-xs">{arm.arm}</td>
                    <td className="px-4 py-3">
                      <p className="font-medium">{arm.label}</p>
                      <p className="text-xs text-muted">{arm.description}</p>
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums">{percent(arm.accuracy, 2)}</td>
                    <td className="px-4 py-3 text-right tabular-nums">{decimal(arm.f1_macro, 4)}</td>
                    <td
                      className={clsx(
                        'px-4 py-3 text-right tabular-nums font-medium',
                        arm.delta_vs_control.f1_macro > 0 ? 'text-leaf-600' : 'text-clay-600',
                      )}
                    >
                      {arm.delta_vs_control.f1_macro === 0
                        ? '—'
                        : `${arm.delta_vs_control.f1_macro > 0 ? '+' : ''}${(
                            arm.delta_vs_control.f1_macro * 100
                          ).toFixed(2)}pt`}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums">{number(arm.parameters)}</td>
                    <td className="px-4 py-3 text-right tabular-nums">{decimal(arm.inference_ms, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {data.transfer_pairs?.available && (
        <Card>
          <CardHeader
            title="A second test: CBAM on pretrained backbones"
            subtitle="Matched pairs from the benchmark — same backbone, with and without CBAM"
            icon={GitCompare}
          />
          <p className="mb-4 text-sm leading-relaxed text-secondary">
            The five-arm ablation answers the research question for one from-scratch CNN.
            The benchmark independently contains the same ImageNet backbone with and
            without a CBAM block, trained under the identical protocol — an independent
            test of the same question in a different regime.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--border-subtle)] bg-[var(--surface-sunken)] text-left">
                  <th className="px-4 py-3 font-medium">Backbone</th>
                  <th className="px-4 py-3 text-right font-medium">Macro F1 without</th>
                  <th className="px-4 py-3 text-right font-medium">Macro F1 with</th>
                  <th className="px-4 py-3 text-right font-medium">Δ</th>
                  <th className="px-4 py-3 text-right font-medium">Params</th>
                  <th className="px-4 py-3 text-right font-medium">Latency</th>
                </tr>
              </thead>
              <tbody>
                {data.transfer_pairs.pairs.map((pair) => (
                  <tr
                    key={pair.backbone}
                    className="border-b border-[var(--border-subtle)] last:border-0"
                  >
                    <td className="px-4 py-2.5 font-medium">{pair.backbone}</td>
                    <td className="px-4 py-2.5 text-right tabular-nums">
                      {decimal(pair.without_cbam.f1_macro, 4)}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums">
                      {decimal(pair.with_cbam.f1_macro, 4)}
                    </td>
                    <td
                      className={clsx(
                        'px-4 py-2.5 text-right font-medium tabular-nums',
                        pair.beyond_noise === false
                          ? 'text-muted'
                          : pair.delta_f1_macro > 0
                            ? 'text-leaf-600'
                            : 'text-clay-600',
                      )}
                    >
                      {`${pair.delta_f1_macro > 0 ? '+' : ''}${(pair.delta_f1_macro * 100).toFixed(2)}pt`}
                      {pair.beyond_noise === false && ' *'}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-muted">
                      {`${pair.parameter_overhead_pct > 0 ? '+' : ''}${pair.parameter_overhead_pct.toFixed(1)}%`}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-muted">
                      {`${pair.inference_overhead_pct > 0 ? '+' : ''}${pair.inference_overhead_pct.toFixed(1)}%`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-sm leading-relaxed">{data.transfer_pairs.verdict}</p>
          <p className="mt-3 rounded-lg bg-[var(--surface-sunken)] p-3 text-sm leading-relaxed text-secondary">
            <strong className="font-medium">Caveat: </strong>
            {data.transfer_pairs.caveat}
            {data.transfer_pairs.significance_threshold_f1 != null && (
              <>
                {' '}An asterisk marks a difference smaller than the measured{' '}
                {(data.transfer_pairs.significance_threshold_f1 * 100).toFixed(2)}-point
                run-to-run variation.
              </>
            )}
          </p>
        </Card>
      )}

      {data.placement?.length > 0 && (
        <Card>
          <CardHeader
            title="CBAM placement"
            subtitle="Where in the network the attention block earns its keep"
            icon={GitCompare}
          />
          <HorizontalBarChart
            data={data.placement.map((row) => ({ ...row }))}
            dataKey="f1_macro"
            height={200}
            formatter={(v) => decimal(v, 4)}
            domain={[
              data.placement.length
                ? Math.max(0, Math.min(...data.placement.map((r) => r.f1_macro ?? 1)) - 0.03)
                : 0,
              1,
            ]}
          />
          {placement?.available && (
            <>
              <div className="mt-4 flex flex-wrap items-center gap-2">
                <span
                  className={clsx(
                    'rounded-full px-2.5 py-1 text-xs font-medium',
                    placement.spread_beyond_noise
                      ? EVIDENCE_BADGE.supported
                      : EVIDENCE_BADGE['within noise'],
                  )}
                >
                  {placement.spread_beyond_noise
                    ? 'Placement changes the result beyond noise'
                    : 'Placement differences are within noise'}
                </span>
                <span className="text-xs text-secondary">
                  spread {(placement.spread_f1_macro * 100).toFixed(2)} pt across{' '}
                  {placement.variants.length} placement
                  {placement.variants.length === 1 ? '' : 's'}
                </span>
              </div>
              <p className="mt-3 text-sm leading-relaxed">{placement.statement}</p>
              {placement.production_replication?.available && (
                <div
                  className={clsx(
                    'mt-3 rounded-lg border p-3',
                    placement.production_replication.agrees_with_fixed_budget
                      ? 'border-leaf-300 dark:border-leaf-800'
                      : 'border-amber-300 dark:border-amber-800',
                  )}
                >
                  <p className="text-xs font-medium uppercase tracking-wide text-muted">
                    Does it survive full-data training?
                  </p>
                  <p className="mt-2 text-sm leading-relaxed">
                    {placement.production_replication.statement}
                  </p>
                </div>
              )}
              <p className="mt-3 rounded-lg bg-[var(--surface-sunken)] p-3 text-sm leading-relaxed text-secondary">
                <strong className="font-medium">Caveat: </strong>
                {placement.caveat}
              </p>
            </>
          )}
        </Card>
      )}
    </div>
  );
}

function Delta({ label, value, invert = false, suffix, raw }) {
  const positive = invert ? value < 0 : value > 0;
  const display = raw != null ? `${raw > 0 ? '+' : ''}${raw.toFixed(1)}${suffix || ''}` : `${value > 0 ? '+' : ''}${(value * 100).toFixed(2)}pt`;
  return (
    <div className="rounded-lg border border-[var(--border-subtle)] p-3">
      <p className="text-xs text-muted">{label}</p>
      <p
        className={clsx(
          'mt-1 text-lg font-semibold tabular-nums',
          Math.abs(value) < 1e-9 ? '' : positive ? 'text-leaf-600' : 'text-clay-600',
        )}
      >
        {display}
      </p>
    </div>
  );
}

/* ----------------------------------------------------------------- training */

function TrainingSection() {
  const { data, loading, error, reload } = useApi(() => api.experiments(), []);
  const [selected, setSelected] = useState(null);

  const experimentId = selected || data?.experiments?.[0]?.experiment_id;
  const { data: curves, loading: curvesLoading } = useApi(
    () => (experimentId ? api.curves(experimentId) : Promise.resolve(null)),
    [experimentId],
  );

  if (loading) return <SkeletonCard lines={8} />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  if (!data?.experiments?.length) {
    return (
      <Card>
        <EmptyState icon={LineChart} title="No experiments logged yet" />
      </Card>
    );
  }

  const history = curves?.history;
  const chartData = (history?.epoch || []).map((epoch, index) => ({
    epoch,
    train_loss: history.train_loss[index],
    val_loss: history.val_loss[index],
    train_acc: history.train_acc[index],
    val_acc: history.val_acc[index],
  }));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader title="Select an experiment" icon={LineChart} />
        <select
          value={experimentId}
          onChange={(event) => setSelected(event.target.value)}
          className="w-full rounded-lg border border-[var(--border-strong)] bg-[var(--surface-raised)] px-3 py-2.5 text-sm"
        >
          {data.experiments.map((experiment) => (
            <option key={experiment.experiment_id} value={experiment.experiment_id}>
              {experiment.label} — {experiment.group} — val {percent(experiment.best_val_accuracy, 2)}
            </option>
          ))}
        </select>

        {(() => {
          const experiment = data.experiments.find((e) => e.experiment_id === experimentId);
          if (!experiment) return null;
          return (
            <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Field label="Best val accuracy" value={percent(experiment.best_val_accuracy, 2)} />
              <Field label="Test accuracy" value={percent(experiment.test_accuracy, 2)} />
              <Field label="Epochs run" value={`${experiment.epochs_run} (best ${experiment.best_epoch})`} />
              <Field label="Training time" value={duration(experiment.training_time_s)} />
              <Field label="Image size" value={`${experiment.config?.image_size}px`} />
              <Field label="Batch size" value={experiment.config?.batch_size} />
              <Field label="Learning rate" value={experiment.config?.learning_rate} />
              <Field label="Device" value={experiment.hardware?.gpu_name || experiment.hardware?.device} />
            </div>
          );
        })()}
      </Card>

      {curvesLoading ? (
        <SkeletonCard lines={6} />
      ) : (
        chartData.length > 0 && (
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader title="Loss" subtitle={curves.label} icon={LineChart} />
              <TrainingCurveChart
                data={chartData}
                series={[
                  { key: 'train_loss', label: 'Train', color: 'var(--color-leaf-600)' },
                  { key: 'val_loss', label: 'Validation', color: 'var(--color-clay-600)' },
                ]}
              />
            </Card>
            <Card>
              <CardHeader title="Accuracy" subtitle={curves.label} icon={LineChart} />
              <TrainingCurveChart
                data={chartData}
                series={[
                  { key: 'train_acc', label: 'Train', color: 'var(--color-leaf-600)' },
                  { key: 'val_acc', label: 'Validation', color: 'var(--color-clay-600)' },
                ]}
                yDomain={[0, 1]}
              />
            </Card>
          </div>
        )
      )}
    </div>
  );
}

/* -------------------------------------------------------------- calibration */

function CalibrationSection() {
  const { data, loading, error, reload } = useApi(() => api.calibration(), []);

  if (loading) return <SkeletonCard lines={8} />;
  if (error) return <ErrorState error={error} onRetry={reload} />;

  const calibration = data.calibration;
  const ood = data.ood;

  return (
    <div className="space-y-6">
      {calibration ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="ECE (raw)" value={decimal(calibration.ece, 4)} sub="Expected calibration error" icon={Gauge} tone="amber" />
            <StatCard
              label="ECE (temperature-scaled)"
              value={decimal(calibration.after_temperature?.ece, 4)}
              sub={`T = ${decimal(calibration.temperature, 3)}`}
              icon={Gauge}
            />
            <StatCard label="Brier score" value={decimal(calibration.brier, 4)} sub="lower is better" icon={Binary} tone="neutral" />
            <StatCard
              label="Confidence gap"
              value={`${(calibration.over_confidence_gap * 100).toFixed(2)}pt`}
              sub="mean confidence − accuracy"
              icon={Gauge}
              tone={Math.abs(calibration.over_confidence_gap) > 0.05 ? 'clay' : 'leaf'}
            />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader
                title="Reliability diagram (raw)"
                subtitle="Points on the diagonal mean perfectly calibrated confidence"
                icon={Gauge}
              />
              <ReliabilityChart bins={calibration.bins} />
            </Card>
            <Card>
              <CardHeader
                title="After temperature scaling"
                subtitle="A single scalar divides the logits; accuracy is unchanged by construction"
                icon={Gauge}
              />
              {calibration.after_temperature ? (
                <ReliabilityChart bins={calibration.after_temperature.bins} />
              ) : (
                <EmptyState icon={Gauge} title="Temperature not fitted" />
              )}
            </Card>
          </div>
        </>
      ) : (
        <Card>
          <EmptyState icon={Gauge} title="No calibration data for the serving model" />
        </Card>
      )}

      <Card>
        <CardHeader
          title="Out-of-distribution detection"
          subtitle="Deciding when the system should decline to answer"
          icon={ShieldAlert}
        />
        {ood ? (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              {Object.entries(ood.metrics?.auroc || {}).map(([name, value]) => (
                <div key={name} className="rounded-lg border border-[var(--border-subtle)] p-3">
                  <p className="text-xs text-muted">{name}</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums">{decimal(value, 4)}</p>
                  <p className="text-xs text-muted">AUROC</p>
                </div>
              ))}
            </div>

            {ood.metrics?.pipeline && (
              <div className="mt-5">
                <p className="mb-2 flex items-center gap-1.5 text-sm font-medium">
                  Whole-pipeline rejection
                  <InfoTooltip text="The image-quality gate runs before the model, so the end-to-end rejection rate is what a user actually experiences. The stand-alone OOD-score AUROC understates the protection." />
                </p>
                <SimpleBarChart
                  data={[
                    { name: 'Quality gate', count: ood.metrics.pipeline.quality_gate_rejects_ood * 100 },
                    { name: 'OOD score', count: ood.metrics.pipeline.ood_score_rejects_ood * 100 },
                    { name: 'Combined', count: ood.metrics.pipeline.pipeline_rejects_ood * 100 },
                    {
                      name: 'False rejects',
                      count: ood.metrics.pipeline.quality_gate_rejects_real_leaves * 100,
                    },
                  ]}
                  dataKey="count"
                  nameKey="name"
                  height={230}
                  formatter={(v) => `${v.toFixed(1)}%`}
                />
                <p className="mt-2 text-sm leading-relaxed text-secondary">
                  {ood.metrics.pipeline.interpretation}
                </p>
              </div>
            )}

            <div className="mt-5 rounded-lg bg-[var(--surface-sunken)] p-3.5 text-sm">
              <p className="font-medium">Thresholds in use</p>
              <p className="mt-1 text-secondary">
                Softmax floor {decimal(ood.selected?.msp_min, 4)} · entropy ceiling{' '}
                {decimal(ood.selected?.entropy_max, 4)} · target true-positive rate{' '}
                {percent(ood.selected?.target_tpr, 0)}
              </p>
              {ood.ood_source?.caveat && (
                <p className="mt-2 text-xs leading-relaxed text-muted">{ood.ood_source.caveat}</p>
              )}
            </div>
          </>
        ) : (
          <EmptyState
            icon={ShieldAlert}
            title="OOD thresholds not calibrated"
            description="Run python training/calibrate_ood.py to measure them."
          />
        )}
      </Card>
    </div>
  );
}

/* -------------------------------------------------------------- performance */

function PerformanceSection() {
  const { data, loading, error, reload } = useApi(() => api.performance(), []);

  if (loading) return <SkeletonCard lines={8} />;
  if (error) return <ErrorState error={error} onRetry={reload} />;

  const serving = data.serving;

  if (!serving) {
    return (
      <Card>
        <EmptyState
          icon={Timer}
          title="Serving performance not measured yet"
          description="Run python scripts/benchmark_serving.py against the exported model."
        />
      </Card>
    );
  }

  const stages = serving.per_request || {};
  const stageOrder = ['quality_ms', 'preprocess_ms', 'inference_ms', 'explain_ms'];
  const stageLabels = {
    quality_ms: 'Image quality',
    preprocess_ms: 'Preprocessing',
    inference_ms: 'Model inference',
    explain_ms: 'Explainability',
  };
  const breakdown = stageOrder
    .filter((key) => stages[key])
    .map((key) => ({ name: stageLabels[key], count: stages[key].mean_ms }));

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Total response"
          value={`${stages.total_ms?.mean_ms.toFixed(0) ?? '—'} ms`}
          sub={`p95 ${stages.total_ms?.p95_ms.toFixed(0) ?? '—'} ms`}
          icon={Timer}
        />
        <StatCard
          label="Model inference"
          value={`${stages.inference_ms?.mean_ms.toFixed(1) ?? '—'} ms`}
          sub={`on ${serving.model.device}`}
          icon={Cpu}
        />
        <StatCard
          label="Throughput"
          value={`${serving.throughput_requests_per_second ?? '—'}/s`}
          sub={serving.explain_enabled ? 'with explainability' : 'without explainability'}
          icon={Zap}
          tone="amber"
        />
        <StatCard
          label="Startup model load"
          value={`${serving.model_loading.cold_load_ms.toFixed(0)} ms`}
          sub="paid once, never per request"
          icon={HardDrive}
          tone="neutral"
        />
      </div>

      <Card>
        <CardHeader
          title="Where the time goes"
          subtitle={`Mean over ${serving.iterations} requests on ${serving.model.label}`}
          icon={Timer}
        />
        <SimpleBarChart
          data={breakdown}
          dataKey="count"
          nameKey="name"
          height={260}
          formatter={(v) => `${v.toFixed(1)} ms`}
        />
        <p className="mt-3 text-sm leading-relaxed text-secondary">{data.note}</p>
      </Card>

      <Card className="p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--border-subtle)] bg-[var(--surface-sunken)] text-left">
                <th className="px-4 py-3 font-medium">Stage</th>
                <th className="px-4 py-3 text-right font-medium">Mean</th>
                <th className="px-4 py-3 text-right font-medium">Median</th>
                <th className="px-4 py-3 text-right font-medium">p95</th>
                <th className="px-4 py-3 text-right font-medium">Max</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(stages).map(([key, stat]) => (
                <tr key={key} className="border-b border-[var(--border-subtle)] last:border-0">
                  <td className="px-4 py-2.5">{stageLabels[key] || 'Total response'}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums">{stat.mean_ms.toFixed(2)} ms</td>
                  <td className="px-4 py-2.5 text-right tabular-nums">{stat.median_ms.toFixed(2)} ms</td>
                  <td className="px-4 py-2.5 text-right tabular-nums">{stat.p95_ms.toFixed(2)} ms</td>
                  <td className="px-4 py-2.5 text-right tabular-nums">{stat.max_ms.toFixed(2)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card>
        <CardHeader
          title="Forward-pass latency by architecture"
          subtitle="Measured during the benchmark, excluding preprocessing"
          icon={Cpu}
        />
        <HorizontalBarChart
          data={data.model_latency.map((row) => ({ ...row }))}
          dataKey="inference_ms"
          height={Math.max(240, data.model_latency.length * 34)}
          formatter={(v) => `${v.toFixed(2)} ms`}
        />
      </Card>

      <Card className="bg-[var(--surface-sunken)]">
        <p className="text-sm leading-relaxed text-secondary">
          <strong className="font-medium text-[var(--text-primary)]">On startup cost. </strong>
          {serving.model_loading.note}
        </p>
      </Card>
    </div>
  );
}

/* ---------------------------------------------------------------- per-class */

function PerClassSection() {
  const { data, loading, error, reload } = useApi(() => api.perClass(), []);

  if (loading) return <SkeletonCard lines={8} />;
  if (error) return <ErrorState error={error} onRetry={reload} title="Per-class metrics unavailable" />;

  const rows = data.classes || [];
  const worst = rows.slice(0, 8);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          title={`Per-class performance — ${data.label}`}
          subtitle={`Evaluated on the ${data.split} split`}
          icon={Grid3x3}
        />
        <p className="mb-4 text-sm text-secondary">
          Sorted by F1 ascending, so the classes the model finds hardest appear first. These are
          where extra data or targeted augmentation would help most.
        </p>
        <HorizontalBarChart
          data={worst.map((row) => ({ label: prettyClass(row.class_name), value: row.f1 }))}
          dataKey="value"
          height={320}
          formatter={(v) => decimal(v, 4)}
          highlightKey="__none"
          domain={[0, 1]}
        />
      </Card>

      <Card className="p-0 overflow-hidden">
        <div className="max-h-[32rem] overflow-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0">
              <tr className="border-b border-[var(--border-subtle)] bg-[var(--surface-sunken)] text-left">
                <th className="px-4 py-3 font-medium">Class</th>
                <th className="px-4 py-3 text-right font-medium">Precision</th>
                <th className="px-4 py-3 text-right font-medium">Recall</th>
                <th className="px-4 py-3 text-right font-medium">F1</th>
                <th className="px-4 py-3 text-right font-medium">Support</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.class_name} className="border-b border-[var(--border-subtle)] last:border-0">
                  <td className="px-4 py-2.5">{prettyClass(row.class_name)}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums">{decimal(row.precision, 4)}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums">{decimal(row.recall, 4)}</td>
                  <td
                    className={clsx(
                      'px-4 py-2.5 text-right tabular-nums font-medium',
                      row.f1 < 0.85 ? 'text-clay-600' : row.f1 < 0.95 ? 'text-amber-600' : 'text-leaf-600',
                    )}
                  >
                    {decimal(row.f1, 4)}
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-muted">{row.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ figures */

function FiguresSection() {
  const { data, loading, error, reload } = useApi(() => api.figures(), []);
  const [group, setGroup] = useState('plots');

  const groups = useMemo(() => Object.keys(data || {}), [data]);

  if (loading) return <SkeletonCard lines={6} />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  if (!groups.length) {
    return (
      <Card>
        <EmptyState icon={Images} title="No figures rendered yet" />
      </Card>
    );
  }

  const active = groups.includes(group) ? group : groups[0];
  const files = data[active] || [];

  return (
    <div className="space-y-5">
      <Tabs
        tabs={groups.map((name) => ({ id: name, label: name.replace(/_/g, ' ') }))}
        active={active}
        onChange={setGroup}
      />
      <p className="text-sm text-secondary">
        {files.length} figure{files.length === 1 ? '' : 's'} rendered by the training pipeline.
      </p>
      <div className="grid gap-5 sm:grid-cols-2">
        {files.map((filename) => (
          <Card key={filename} className="p-3">
            <a href={figureUrl(active, filename)} target="_blank" rel="noreferrer">
              <img
                src={figureUrl(active, filename)}
                alt={filename}
                loading="lazy"
                className="w-full rounded-lg border border-[var(--border-subtle)] bg-white"
              />
            </a>
            <p className="mt-2 truncate font-mono text-xs text-muted">{filename}</p>
          </Card>
        ))}
      </div>
    </div>
  );
}

function Field({ label, value }) {
  return (
    <div className="rounded-lg border border-[var(--border-subtle)] p-3">
      <p className="text-xs text-muted">{label}</p>
      <p className="mt-1 truncate font-semibold tabular-nums">{value ?? '—'}</p>
    </div>
  );
}
