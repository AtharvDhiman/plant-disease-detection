import clsx from 'clsx';
import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Gauge,
  HelpCircle,
  ImageIcon,
  Leaf,
  Timer,
  TrendingUp,
} from 'lucide-react';

import { imageUrl } from '../services/api';
import { CONFIDENCE_STYLES, milliseconds, percent, prettyClass } from '../utils/format';
import DiseaseInfoPanel from './DiseaseInfoPanel';
import ExplainabilityViewer from './ExplainabilityViewer';
import { Card, CardHeader, Chip, ProgressBar } from './ui';

const STATUS_BANNERS = {
  poor_quality: {
    icon: AlertTriangle,
    tone: 'border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-100',
    title: 'Image quality is insufficient for a reliable prediction',
  },
  out_of_distribution: {
    icon: HelpCircle,
    tone: 'border-clay-300 bg-clay-50 text-clay-900 dark:border-clay-800 dark:bg-clay-950/60 dark:text-clay-100',
    title: 'Unable to confidently identify this image',
  },
  low_confidence: {
    icon: AlertTriangle,
    tone: 'border-clay-300 bg-clay-50 text-clay-900 dark:border-clay-800 dark:bg-clay-950/60 dark:text-clay-100',
    title: 'Low confidence prediction',
  },
};

function StatusBanner({ status, message }) {
  const config = STATUS_BANNERS[status];
  if (!config) return null;
  const Icon = config.icon;
  return (
    <div className={clsx('flex items-start gap-3 rounded-xl border p-4', config.tone)}>
      <Icon size={19} className="mt-0.5 shrink-0" />
      <div>
        <p className="font-semibold">{config.title}</p>
        {message && <p className="mt-1 text-sm opacity-90">{message}</p>}
      </div>
    </div>
  );
}

export default function PredictionResult({ result }) {
  // A refused analysis must not wear a green "High confidence" chip. The
  // reported confidence is temperature-scaled, while the reliability decision is
  // taken on the raw logits the thresholds were measured on - so a leaf can read
  // 94.7% here and still sit in the lowest 5% of genuine leaves by the
  // uncalibrated measure that actually governs the verdict. Both numbers are
  // correct on their own scale; showing them side by side without saying so is
  // what made the page look self-contradictory.
  const refused = result.status && result.status !== 'ok';
  const confidence = refused
    ? {
        chip: 'border-clay-300 bg-clay-50 text-clay-900 '
          + 'dark:border-clay-800 dark:bg-clay-950/60 dark:text-clay-100',
        dot: 'bg-clay-500',
        label: 'Not accepted',
      }
    : CONFIDENCE_STYLES[result.confidence_level] || CONFIDENCE_STYLES.low;
  const original = result.explanation?.images?.original;
  const quality = result.quality;

  return (
    <div className="space-y-6 animate-fade-in-up">
      <StatusBanner status={result.status} message={result.status_message} />

      {/* ------------------------------------------------------ headline card */}
      <Card className="overflow-hidden p-0">
        <div className="grid lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
          <div className="bg-black/5 p-4 dark:bg-black/30">
            {original ? (
              <img
                src={imageUrl(original)}
                alt="Analysed leaf"
                className="mx-auto w-full rounded-xl object-contain"
              />
            ) : (
              <div className="flex h-52 items-center justify-center rounded-xl bg-[var(--surface-sunken)] text-muted">
                <ImageIcon size={30} />
              </div>
            )}
          </div>

          <div className="p-5 sm:p-6">
            <div className="flex flex-wrap items-center gap-2">
              <Chip className={confidence.chip}>
                <span className={clsx('h-1.5 w-1.5 rounded-full', confidence.dot)} />
                {confidence.label}
              </Chip>
              {result.is_healthy ? (
                <Chip className="border-leaf-300 bg-leaf-100 text-leaf-800 dark:border-leaf-800 dark:bg-leaf-900/60 dark:text-leaf-200">
                  <CheckCircle2 size={12} />
                  Healthy
                </Chip>
              ) : (
                <Chip className="border-clay-300 bg-clay-100 text-clay-800 dark:border-clay-800 dark:bg-clay-900/60 dark:text-clay-200">
                  Disease detected
                </Chip>
              )}
            </div>

            <p className="mt-4 text-sm font-medium uppercase tracking-wide text-muted">
              {result.plant}
            </p>
            <h2 className="mt-1 text-3xl font-semibold leading-tight tracking-tight">
              {result.condition}
            </h2>
            <p className="mt-1 font-mono text-xs text-muted">{result.predicted_class}</p>

            <div className="mt-5">
              <div className="mb-2 flex items-baseline justify-between">
                <span className="text-sm text-secondary">Confidence</span>
                <span className="text-2xl font-semibold tabular-nums">
                  {percent(result.confidence, 2)}
                </span>
              </div>
              <ProgressBar
                value={result.confidence}
                barClassName={confidence.bar}
                label="Prediction confidence"
              />
            </div>

            <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Metric icon={Cpu} label="Model" value={result.model_info.label} small />
              <Metric icon={Timer} label="Inference" value={milliseconds(result.timings.inference_ms)} />
              <Metric icon={Gauge} label="Total" value={milliseconds(result.timings.total_ms)} />
              <Metric icon={ImageIcon} label="Image quality" value={percent(quality.score, 0)} />
            </div>
          </div>
        </div>
      </Card>

      {/* -------------------------------------------------- top-k predictions */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader
            title="Top predictions"
            subtitle="Calibrated class probabilities"
            icon={TrendingUp}
          />
          <ul className="space-y-3">
            {result.top_predictions.map((item, index) => (
              <li key={item.class_name}>
                <div className="mb-1.5 flex items-baseline justify-between gap-3">
                  <span className={clsx('truncate text-sm', index === 0 ? 'font-semibold' : 'text-secondary')}>
                    <span className="mr-2 text-muted tabular-nums">{index + 1}.</span>
                    {prettyClass(item.class_name)}
                  </span>
                  <span className="shrink-0 text-sm font-medium tabular-nums">
                    {percent(item.probability, 2)}
                  </span>
                </div>
                <ProgressBar
                  value={item.probability}
                  barClassName={index === 0 ? confidence.bar : 'bg-[var(--border-strong)]'}
                />
              </li>
            ))}
          </ul>
        </Card>

        <Card>
          <CardHeader
            title="Reliability checks"
            subtitle="What the system verified before answering"
            icon={Leaf}
          />
          <dl className="space-y-3 text-sm">
            <CheckRow
              ok={quality.passed}
              label="Image quality"
              detail={
                quality.issues.length
                  ? quality.issues.map((issue) => issue.message).join(' ')
                  : `Score ${percent(quality.score, 0)} — sharp, well exposed and plant-like.`
              }
            />
            <CheckRow
              ok={!result.ood.is_out_of_distribution}
              label="In-distribution check"
              detail={
                result.ood.is_out_of_distribution
                  ? result.ood.reasons.join(' ')
                  : `Softmax ${percent(result.ood.scores.msp, 1)}, normalised entropy ${result.ood.scores.entropy?.toFixed(
                      2,
                    )} — consistent with the training distribution.`
              }
            />
            <CheckRow
              ok={!refused && result.confidence_level !== 'low'}
              label="Confidence threshold"
              detail={`${percent(result.confidence, 1)} — classified as ${result.confidence_level}.`}
            />
          </dl>

          {quality.issues.length > 0 && (
            <div className="mt-4 rounded-lg bg-[var(--surface-sunken)] p-3">
              <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-muted">
                How to improve the photo
              </p>
              <ul className="space-y-1 text-sm text-secondary">
                {quality.issues.map((issue) => (
                  <li key={issue.code}>• {issue.suggestion}</li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      </div>

      <ExplainabilityViewer explanation={result.explanation} />
      <DiseaseInfoPanel info={result.disease_info} />
    </div>
  );
}

function Metric({ icon: Icon, label, value, small }) {
  return (
    <div className="rounded-lg border border-[var(--border-subtle)] p-2.5">
      <p className="flex items-center gap-1.5 text-xs text-muted">
        <Icon size={12} />
        {label}
      </p>
      <p className={clsx('mt-1 font-semibold tabular-nums truncate', small ? 'text-xs' : 'text-sm')}>
        {value}
      </p>
    </div>
  );
}

function CheckRow({ ok, label, detail }) {
  return (
    <div className="flex items-start gap-2.5">
      {ok ? (
        <CheckCircle2 size={16} className="mt-0.5 shrink-0 text-leaf-600" />
      ) : (
        <AlertTriangle size={16} className="mt-0.5 shrink-0 text-clay-600" />
      )}
      <div className="min-w-0">
        <dt className="font-medium">{label}</dt>
        <dd className="text-secondary leading-relaxed">{detail}</dd>
      </div>
    </div>
  );
}
