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
  not_a_plant: {
    icon: AlertTriangle,
    tone: 'border-red-300 bg-red-50 text-red-900 dark:border-red-800 dark:bg-red-950/60 dark:text-red-100',
    title: 'Photo Rejected: Not a recognized plant leaf',
  },
  poor_quality: {
    icon: AlertTriangle,
    tone: 'border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-100',
    title: 'Image quality is insufficient for a reliable prediction',
  },
  out_of_distribution: {
    icon: HelpCircle,
    tone: 'border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-100',
    title: 'Advisory: Atypical leaf pattern or moderate confidence',
  },
  low_confidence: {
    icon: AlertTriangle,
    tone: 'border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-100',
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
  const isRejected =
    result.status === 'not_a_plant' ||
    result.status === 'poor_quality';

  const confidence = isRejected
    ? {
        chip:
          'border-clay-300 bg-clay-50 text-clay-900 ' +
          'dark:border-clay-800 dark:bg-clay-950/60 dark:text-clay-100',
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
                alt="Analysed upload"
                className="mx-auto w-full rounded-xl object-contain"
              />
            ) : (
              <div className="flex h-52 items-center justify-center rounded-xl bg-[var(--surface-sunken)] text-muted">
                <ImageIcon size={30} />
              </div>
            )}
          </div>

          <div className="p-5 sm:p-6">
            {isRejected ? (
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <Chip className="border-red-300 bg-red-100 text-red-800 dark:border-red-800 dark:bg-red-900/60 dark:text-red-200">
                    <AlertTriangle size={12} />
                    Photo Rejected
                  </Chip>
                  <Chip className="border-clay-300 bg-clay-100 text-clay-800 dark:border-clay-800 dark:bg-clay-900/60 dark:text-clay-200">
                    No Diagnosis Provided
                  </Chip>
                </div>

                <h2 className="mt-3 text-2xl font-semibold leading-tight tracking-tight text-[var(--text-primary)]">
                  {result.status === 'not_a_plant'
                    ? 'No Plant Leaf Detected'
                    : 'Insufficient Image Quality'}
                </h2>

                <p className="mt-2 text-sm text-secondary leading-relaxed">
                  {result.status_message ||
                    'This image was rejected by our input verification gate. Only genuine leaves of supported agricultural crops are diagnosed.'}
                </p>

                <div className="mt-4 rounded-xl border border-[var(--border-subtle)] bg-[var(--surface-sunken)] p-3.5">
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted">
                    Supported Agricultural Crops (14 species)
                  </p>
                  <p className="mt-1 text-xs text-secondary leading-relaxed">
                    Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper, Potato,
                    Raspberry, Soybean, Squash, Strawberry, and Tomato.
                  </p>
                </div>

                <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Metric icon={Cpu} label="Model" value={result.model_info?.label || 'EfficientNet'} small />
                  <Metric icon={Timer} label="Inference" value={result.timings?.inference_ms != null ? milliseconds(result.timings.inference_ms) : '—'} />
                  <Metric icon={Gauge} label="Total" value={result.timings?.total_ms != null ? milliseconds(result.timings.total_ms) : '—'} />
                  <Metric icon={ImageIcon} label="Image quality" value={quality?.score != null ? percent(quality.score, 0) : '—'} />
                </div>
              </div>
            ) : (
              <div>
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
                  <Metric icon={Cpu} label="Model" value={result.model_info?.label || 'EfficientNet'} small />
                  <Metric icon={Timer} label="Inference" value={result.timings?.inference_ms != null ? milliseconds(result.timings.inference_ms) : '—'} />
                  <Metric icon={Gauge} label="Total" value={result.timings?.total_ms != null ? milliseconds(result.timings.total_ms) : '—'} />
                  <Metric icon={ImageIcon} label="Image quality" value={quality?.score != null ? percent(quality.score, 0) : '—'} />
                </div>
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* -------------------------------------------------- middle section */}
      <div className="grid gap-6 lg:grid-cols-2">
        {isRejected ? (
          <Card>
            <CardHeader
              title="Verification details"
              subtitle="Why this photo was refused"
              icon={AlertTriangle}
            />
            <div className="space-y-3 text-sm">
              <p className="text-secondary leading-relaxed">
                The model is trained solely to diagnose diseases on leaves of 14 agricultural crops.
                Arbitrary non-plant images (portraits, cars, landscapes, pets, or non-leaf items) are
                rejected to prevent dangerous false diagnoses.
              </p>
              <div className="rounded-lg bg-[var(--surface-sunken)] p-3 text-xs text-secondary space-y-1">
                <p className="font-medium text-[var(--text-primary)]">To get an accurate result:</p>
                <p>• Ensure the subject is a leaf of a supported crop.</p>
                <p>• Fill at least 30–50% of the frame with the leaf.</p>
                <p>• Avoid blurry, shaky, dark, or heavily back-lit shots.</p>
              </div>
            </div>
          </Card>
        ) : (
          <Card>
            <CardHeader
              title="Top predictions"
              subtitle="Calibrated class probabilities"
              icon={TrendingUp}
            />
            <ul className="space-y-3">
              {(result.top_predictions || []).map((item, index) => (
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
        )}

        <Card>
          <CardHeader
            title="Reliability checks"
            subtitle="What the system verified before answering"
            icon={Leaf}
          />
          <dl className="space-y-3 text-sm">
            <CheckRow
              ok={Boolean(quality?.passed)}
              label="Image quality & plant detection"
              detail={
                quality?.issues?.length
                  ? quality.issues.map((issue) => issue.message).join(' ')
                  : `Score ${percent(quality?.score ?? 0, 0)} — sharp, well exposed and plant-like.`
              }
            />
            <CheckRow
              ok={!result.ood?.is_out_of_distribution}
              label="In-distribution check"
              detail={
                result.ood?.is_out_of_distribution
                  ? (result.ood.reasons || []).join(' ')
                  : `Softmax ${percent(result.ood?.scores?.msp ?? 0, 1)}, normalised entropy ${result.ood?.scores?.entropy != null ? result.ood.scores.entropy.toFixed(2) : '—'} — consistent with the training distribution.`
              }
            />
            <CheckRow
              ok={!isRejected && result.confidence_level !== 'low'}
              label="Confidence threshold"
              detail={
                isRejected
                  ? 'Analysis refused by validation safety filters.'
                  : `${percent(result.confidence, 1)} — classified as ${result.confidence_level}.`
              }
            />
          </dl>

          {(quality?.issues || []).length > 0 && (
            <div className="mt-4 rounded-lg bg-[var(--surface-sunken)] p-3">
              <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-muted">
                How to improve the photo
              </p>
              <ul className="space-y-1 text-sm text-secondary">
                {(quality?.issues || []).map((issue) => (
                  <li key={issue.code}>• {issue.suggestion}</li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      </div>

      {!isRejected && result.explanation && Object.keys(result.explanation.images || {}).length > 0 && (
        <ExplainabilityViewer explanation={result.explanation} />
      )}
      {!isRejected && result.disease_info && (
        <DiseaseInfoPanel info={result.disease_info} />
      )}
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
