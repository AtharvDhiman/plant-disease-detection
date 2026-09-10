import {
  Activity,
  BrainCircuit,
  Camera,
  Layers,
  ScanLine,
  ShieldCheck,
  Sparkles,
  Zap,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import ImageUploader from '../components/ImageUploader';
import PredictionResult from '../components/PredictionResult';
import { Button, Card } from '../components/ui';
import { useToast } from '../components/ui/toast-context';
import { api } from '../services/api';
import { decimal, percent } from '../utils/format';

const FEATURES = [
  {
    icon: BrainCircuit,
    title: 'CNN with CBAM attention',
    body: 'Channel attention learns which features matter; spatial attention learns where on the leaf they matter. Both are applied in sequence.',
  },
  {
    icon: Layers,
    title: 'Explains every decision',
    body: 'Grad-CAM, Grad-CAM++ and the network’s own learned attention map show which region drove the prediction.',
  },
  {
    icon: ShieldCheck,
    title: 'Knows when to abstain',
    body: 'Image-quality checks and an out-of-distribution score mean a blurry photo or a non-leaf image is reported, not guessed at.',
  },
  {
    icon: Activity,
    title: 'Benchmarked honestly',
    body: 'Every architecture was trained under one identical protocol. The numbers on this site come straight from those runs.',
  },
];

function HeroStats() {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    let active = true;
    Promise.all([api.models().catch(() => null), api.datasetStats().catch(() => null)]).then(
      ([models, dataset]) => {
        if (!active) return;
        setStats({ models, dataset });
      },
    );
    return () => {
      active = false;
    };
  }, []);

  const metrics = stats?.models?.production?.test_metrics;
  const items = [
    {
      label: 'Test accuracy',
      value: metrics?.accuracy != null ? percent(metrics.accuracy, 2) : '—',
      hint: stats?.models?.production?.label,
    },
    {
      label: 'Macro F1',
      value: metrics?.f1_macro != null ? decimal(metrics.f1_macro, 4) : '—',
      hint: 'Every class weighted equally',
    },
    {
      label: 'Disease classes',
      value: stats?.dataset?.num_classes ?? '—',
      hint: `${stats?.dataset?.num_plants ?? '—'} plant species`,
    },
    {
      label: 'Training images',
      value: stats?.dataset?.total_images ? stats.dataset.total_images.toLocaleString() : '—',
      hint: 'PlantVillage (augmented)',
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {items.map((item) => (
        <div
          key={item.label}
          className="rounded-xl border border-[var(--border-subtle)] bg-[var(--surface-raised)]/70 p-3.5 backdrop-blur"
        >
          <p className="text-xs text-muted">{item.label}</p>
          <p className="mt-1 text-xl font-semibold tabular-nums">{item.value}</p>
          {item.hint && <p className="mt-0.5 truncate text-xs text-muted">{item.hint}</p>}
        </div>
      ))}
    </div>
  );
}

export default function Home() {
  const navigate = useNavigate();
  const [result, setResult] = useState(null);
  const [analysing, setAnalysing] = useState(false);
  const uploadRef = useRef(null);
  const resultRef = useRef(null);
  const toast = useToast();

  const analyse = useCallback(
    async (file) => {
      if (!file) return;
      setAnalysing(true);
      setResult(null);
      try {
        const prediction = await api.predict(file, { topK: 5, explain: true, save: true });
        setResult(prediction);

        if (prediction.status === 'ok' && prediction.is_healthy) {
          toast.push('No disease detected in this leaf.', { type: 'success', title: 'Healthy' });
        } else if (prediction.status === 'ok') {
          toast.push(`${prediction.display_name} — ${percent(prediction.confidence, 1)} confidence`, {
            type: 'warning',
            title: 'Disease detected',
          });
        } else if (prediction.status === 'not_a_plant') {
          toast.push('Photo rejected: No plant leaf detected in this image.', {
            type: 'error',
            title: 'Not a plant leaf',
          });
        } else if (prediction.status === 'out_of_distribution') {
          toast.push(
            `${prediction.display_name} (${percent(prediction.confidence, 1)}) — leaf pattern is atypical or confidence is moderate`,
            {
              type: 'warning',
              title: 'Diagnosis advisory',
            },
          );
        } else {
          toast.push(prediction.status_message || 'The result needs review.', {
            type: 'warning',
            title: 'Please review',
          });
        }
        // Let React paint the result before scrolling to it.
        requestAnimationFrame(() =>
          resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
        );
      } catch (error) {
        toast.push(error.hint ? `${error.message} ${error.hint}` : error.message, {
          type: 'error',
          title: 'Analysis failed',
          duration: 8000,
        });
      } finally {
        setAnalysing(false);
      }
    },
    [toast],
  );

  return (
    <div>
      {/* ------------------------------------------------------------- hero */}
      <section className="relative overflow-hidden border-b border-[var(--border-subtle)]">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-[0.55] dark:opacity-40"
          style={{
            backgroundImage:
              'radial-gradient(60rem 30rem at 15% -10%, var(--color-leaf-200), transparent 65%), radial-gradient(45rem 25rem at 95% 10%, var(--color-leaf-100), transparent 60%)',
          }}
        />
        <div className="relative mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:py-20">
          <div className="grid items-center gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,30rem)]">
            <div>
              <span className="inline-flex items-center gap-2 rounded-full border border-leaf-300 bg-leaf-50 px-3 py-1 text-xs font-medium text-leaf-800 dark:border-leaf-800 dark:bg-leaf-950/60 dark:text-leaf-200">
                <Sparkles size={13} />
                Deep learning with attention mechanisms
              </span>

              <h1 className="mt-5 text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl lg:text-6xl">
                AI-Powered Plant
                <br />
                <span className="text-leaf-600">Disease Detection</span>
              </h1>

              <p className="mt-5 max-w-xl text-lg leading-relaxed text-secondary">
                Detect plant diseases from leaf images using deep learning and attention
                mechanisms. Upload a photo and get a calibrated diagnosis, a visual explanation of
                what the model looked at, and practical guidance.
              </p>

              <div className="mt-7 flex flex-wrap gap-3">
                <Button
                  size="lg"
                  icon={ScanLine}
                  onClick={() => uploadRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })}
                >
                  Analyse a leaf
                </Button>
                <Button
                  size="lg"
                  variant="secondary"
                  icon={Activity}
                  onClick={() => {
                    navigate('/benchmark');
                  }}
                >
                  See the benchmark
                </Button>
              </div>

              <div className="mt-9">
                <HeroStats />
              </div>
            </div>

            <div ref={uploadRef} className="lg:sticky lg:top-24">
              <Card className="p-5 sm:p-6">
                <div className="mb-4 flex items-center gap-2.5">
                  <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-leaf-100 text-leaf-700 dark:bg-leaf-900/60 dark:text-leaf-300">
                    <Camera size={18} />
                  </span>
                  <div>
                    <h2 className="font-semibold leading-tight">Upload a leaf image</h2>
                    <p className="text-sm text-secondary">One leaf, filling most of the frame</p>
                  </div>
                </div>
                <ImageUploader onAnalyse={analyse} analysing={analysing} />
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* ----------------------------------------------------------- result */}
      <div ref={resultRef} className="mx-auto max-w-7xl px-4 sm:px-6">
        {result && (
          <section className="py-10">
            <h2 className="mb-6 text-2xl font-semibold tracking-tight">Analysis result</h2>
            <PredictionResult result={result} />
          </section>
        )}
      </div>

      {/* --------------------------------------------------------- features */}
      {!result && (
        <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
          <div className="mb-10 max-w-2xl">
            <h2 className="text-3xl font-semibold tracking-tight">
              Built like a diagnostic tool, not a demo
            </h2>
            <p className="mt-3 text-secondary">
              A classifier that is confidently wrong is worse than no classifier. This system
              measures its own inputs, explains its outputs and reports honestly when it does not
              know.
            </p>
          </div>

          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {FEATURES.map(({ icon: Icon, title, body }) => (
              <Card key={title} className="transition-shadow hover:shadow-lg">
                <span className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-leaf-100 text-leaf-700 dark:bg-leaf-900/60 dark:text-leaf-300">
                  <Icon size={20} />
                </span>
                <h3 className="font-semibold">{title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-secondary">{body}</p>
              </Card>
            ))}
          </div>

          {/* ------------------------------------------------ how it works */}
          <div className="mt-16">
            <h2 className="text-2xl font-semibold tracking-tight">How a prediction is made</h2>
            <ol className="mt-6 grid gap-4 md:grid-cols-3 lg:grid-cols-6">
              {[
                { icon: Camera, title: 'Upload', body: 'A leaf photo from a phone or camera.' },
                { icon: ShieldCheck, title: 'Quality check', body: 'Sharpness, exposure, contrast and plant coverage.' },
                { icon: Zap, title: 'Preprocess', body: 'Resize and normalise exactly as in training.' },
                { icon: BrainCircuit, title: 'CNN + CBAM', body: 'Channel then spatial attention over the features.' },
                { icon: Layers, title: 'Explain', body: 'Grad-CAM and the learned attention map.' },
                { icon: Activity, title: 'Report', body: 'Calibrated confidence and disease guidance.' },
              ].map(({ icon: Icon, title, body }, index) => (
                <li key={title} className="relative rounded-xl border border-[var(--border-subtle)] p-4">
                  <span className="absolute -top-2.5 left-4 rounded-full bg-leaf-600 px-2 py-0.5 text-[11px] font-semibold text-white">
                    {index + 1}
                  </span>
                  <Icon size={19} className="mt-1 text-leaf-600" />
                  <p className="mt-2.5 font-medium">{title}</p>
                  <p className="mt-1 text-xs leading-relaxed text-secondary">{body}</p>
                </li>
              ))}
            </ol>
          </div>
        </section>
      )}
    </div>
  );
}
