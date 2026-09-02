import clsx from 'clsx';
import { AlertTriangle, Eye, Info, Layers, Sparkles } from 'lucide-react';
import { useState } from 'react';

import { imageUrl } from '../services/api';
import { percent } from '../utils/format';
import { Card, CardHeader, InfoTooltip, Tabs } from './ui';

const METHOD_META = {
  original: {
    label: 'Original',
    icon: Eye,
    blurb:
      'The image exactly as the model sees it, after resizing and centre-cropping. Comparing this with the heat maps is the point: the crop is what the network actually had to work with.',
  },
  grad_cam: {
    label: 'Grad-CAM',
    icon: Layers,
    blurb:
      'Weights the final convolutional feature maps by how much increasing each one would raise the score for the predicted class. Bright regions are where the evidence for the decision was pooled from.',
  },
  grad_cam_plus_plus: {
    label: 'Grad-CAM++',
    icon: Layers,
    blurb:
      'A refinement of Grad-CAM that uses higher-order gradient terms, so several separate lesions all contribute instead of the single strongest one dominating.',
  },
  cbam_spatial: {
    label: 'CBAM attention',
    icon: Sparkles,
    blurb:
      'Not a saliency method at all: this is the spatial gate the network itself learned and applied during the forward pass. It shows where CBAM chose to amplify the features.',
  },
  integrated_gradients: {
    label: 'Integrated gradients',
    icon: Layers,
    blurb:
      'Pixel-level attribution obtained by integrating gradients along a path from a black baseline to the real image. Noisier than Grad-CAM but it satisfies a completeness guarantee that Grad-CAM does not.',
  },
};

export default function ExplainabilityViewer({ explanation }) {
  const images = explanation?.images || {};
  const order = ['original', 'grad_cam', 'grad_cam_plus_plus', 'cbam_spatial', 'integrated_gradients'];
  const available = order.filter((key) => images[key]);
  const [active, setActive] = useState(available.includes('grad_cam') ? 'grad_cam' : available[0]);

  if (available.length === 0) {
    const failed = Object.entries(explanation?.errors || {});
    return (
      <Card>
        <CardHeader title="Explainability" icon={Layers} />
        {failed.length ? (
          <div className="text-sm">
            <p className="font-medium text-clay-700 dark:text-clay-300">
              Explanations were requested but could not be generated.
            </p>
            <ul className="mt-2 space-y-1 text-secondary">
              {failed.map(([method, reason]) => (
                <li key={method}>
                  <span className="font-medium">{METHOD_META[method]?.label || method}:</span>{' '}
                  {reason}
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <p className="text-sm text-secondary">
            Heat maps were not generated for this prediction.
          </p>
        )}
      </Card>
    );
  }

  const meta = METHOD_META[active] || { label: active, blurb: '' };
  const stats = explanation.stats?.[active];
  const channels = explanation.channel_attention;
  const failures = Object.entries(explanation.errors || {});

  return (
    <Card>
      <CardHeader
        title="Explainability"
        subtitle="Where the model looked to reach this decision"
        icon={Layers}
      />

      <Tabs
        tabs={available.map((key) => ({ id: key, label: METHOD_META[key]?.label || key }))}
        active={active}
        onChange={setActive}
        className="mb-4"
      />

      {failures.length > 0 && (
        <div className="mb-4 flex items-start gap-2.5 rounded-xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-100">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-medium">
              {failures.length === 1
                ? 'One explanation could not be generated'
                : `${failures.length} explanations could not be generated`}
            </p>
            <ul className="mt-1 space-y-0.5 text-xs opacity-90">
              {failures.map(([method, reason]) => (
                <li key={method}>
                  <span className="font-medium">{METHOD_META[method]?.label || method}:</span>{' '}
                  {reason}
                </li>
              ))}
            </ul>
            <p className="mt-1.5 text-xs opacity-80">
              The diagnosis itself is unaffected — a heat map never blocks a prediction.
            </p>
          </div>
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <div className="overflow-hidden rounded-xl border border-[var(--border-subtle)] bg-black/5 dark:bg-black/30">
          <img
            src={imageUrl(images[active])}
            alt={`${meta.label} visualisation`}
            className="mx-auto w-full object-contain"
            loading="lazy"
          />
        </div>

        <div className="space-y-4">
          <div className="rounded-xl bg-[var(--surface-sunken)] p-3.5">
            <p className="flex items-center gap-2 text-sm font-medium">
              <Info size={15} className="text-leaf-600" />
              {meta.label}
            </p>
            <p className="mt-1.5 text-sm text-secondary leading-relaxed">{meta.blurb}</p>
          </div>

          {active !== 'original' && (
            <div>
              <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted">Colour scale</p>
              <div
                className="h-3 w-full rounded-full"
                style={{
                  backgroundImage:
                    'linear-gradient(90deg,#000080,#0000ff,#0080ff,#00ffff,#80ff80,#ffff00,#ff8000,#ff0000,#800000)',
                }}
              />
              <div className="mt-1 flex justify-between text-xs text-muted">
                <span>Low influence</span>
                <span>High influence</span>
              </div>
            </div>
          )}

          {stats && (
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border border-[var(--border-subtle)] p-3">
                <p className="flex items-center gap-1 text-xs text-muted">
                  Focus area
                  <InfoTooltip text="Fraction of the image the heat map marks as strongly influential. A very small value means the decision rested on a small patch; a very large one can indicate the model used the background." />
                </p>
                <p className="mt-1 text-lg font-semibold tabular-nums">{percent(stats.focus_ratio, 1)}</p>
              </div>
              <div className="rounded-lg border border-[var(--border-subtle)] p-3">
                <p className="text-xs text-muted">Mean intensity</p>
                <p className="mt-1 text-lg font-semibold tabular-nums">{percent(stats.mean_value, 1)}</p>
              </div>
            </div>
          )}

          {active === 'cbam_spatial' && channels?.length > 0 && (
            <div>
              <p className="mb-2 flex items-center gap-1 text-xs font-medium uppercase tracking-wide text-muted">
                CBAM channel gates
                <InfoTooltip text="The channel-attention branch produces one gate value per feature channel, deciding WHAT to emphasise. Each bar is one channel of the deepest CBAM block." />
              </p>
              <div className="flex h-14 items-end gap-px overflow-hidden rounded-lg bg-[var(--surface-sunken)] p-1.5">
                {channels.slice(0, 96).map((value, index) => (
                  <div
                    key={index}
                    className="flex-1 rounded-sm bg-leaf-500"
                    style={{ height: `${Math.max(4, value * 100)}%`, opacity: 0.45 + value * 0.55 }}
                    title={`Channel ${index}: ${value.toFixed(3)}`}
                  />
                ))}
              </div>
              <p className="mt-1.5 text-xs text-muted">
                Showing the first {Math.min(96, channels.length)} of {channels.length} channels.
              </p>
            </div>
          )}
        </div>
      </div>

      <p className={clsx('mt-4 rounded-lg border border-[var(--border-subtle)] p-3 text-xs leading-relaxed text-muted')}>
        <strong className="font-medium text-[var(--text-secondary)]">How to read this: </strong>
        {explanation.caveat ||
          'Heat maps show where the network pooled evidence for its decision, at the resolution of its final feature map. They are not a disease segmentation and do not measure lesion extent.'}{' '}
        If the highlighted region sits on the background rather than the leaf, treat the prediction
        with caution.
      </p>
    </Card>
  );
}
