import clsx from 'clsx';
import { AlertTriangle, Camera, CheckCircle2, ImageIcon, Loader2, Upload, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';

import { api } from '../services/api';
import { Button } from './ui';

const ACCEPTED = 'image/jpeg,image/png,image/webp';
const MAX_BYTES = 10 * 1024 * 1024;

/**
 * Drag-and-drop / browse / camera capture with an immediate quality pre-check.
 *
 * The quality check runs as soon as a file is chosen, before the user commits
 * to a full analysis, so a bad photo can be replaced without waiting for
 * inference and explainability to finish.
 */
export default function ImageUploader({ onAnalyse, analysing, disabled }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);
  const [quality, setQuality] = useState(null);
  const [checking, setChecking] = useState(false);
  const inputRef = useRef(null);
  const cameraRef = useRef(null);

  // Object URLs must be revoked or the blob stays in memory for the session.
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  const accept = useCallback(async (chosen) => {
    setError(null);
    setQuality(null);

    if (!chosen) return;
    if (!ACCEPTED.split(',').includes(chosen.type)) {
      setError(`"${chosen.type || 'unknown type'}" is not supported. Use JPG, PNG or WEBP.`);
      return;
    }
    if (chosen.size > MAX_BYTES) {
      setError(`That file is ${(chosen.size / 1024 ** 2).toFixed(1)} MB; the limit is 10 MB.`);
      return;
    }

    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(chosen);
    });
    setFile(chosen);

    setChecking(true);
    try {
      const report = await api.analyzeImage(chosen);
      setQuality(report);
    } catch (err) {
      // A failed pre-check should not block the user from trying a full analysis.
      setQuality(null);
      setError(err.message);
    } finally {
      setChecking(false);
    }
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();
      setDragging(false);
      accept(event.dataTransfer.files?.[0]);
    },
    [accept],
  );

  const reset = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setQuality(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  const band = quality?.quality?.band;
  const bandStyles = {
    good: 'border-leaf-300 bg-leaf-50 text-leaf-900 dark:bg-leaf-950/60 dark:text-leaf-100 dark:border-leaf-800',
    acceptable:
      'border-amber-300 bg-amber-50 text-amber-900 dark:bg-amber-950/60 dark:text-amber-100 dark:border-amber-800',
    poor: 'border-clay-300 bg-clay-50 text-clay-900 dark:bg-clay-950/60 dark:text-clay-100 dark:border-clay-800',
  };

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        className="hidden"
        onChange={(event) => accept(event.target.files?.[0])}
      />
      {/* `capture` asks a mobile browser for the rear camera directly. */}
      <input
        ref={cameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={(event) => accept(event.target.files?.[0])}
      />

      {!preview ? (
        <div
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault();
              inputRef.current?.click();
            }
          }}
          role="button"
          tabIndex={0}
          aria-label="Upload a leaf image"
          className={clsx(
            'flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-14 text-center transition-colors',
            dragging
              ? 'border-leaf-500 bg-leaf-50 dark:bg-leaf-950/50'
              : 'border-[var(--border-strong)] hover:border-leaf-400 hover:bg-[var(--surface-sunken)]',
          )}
        >
          <span
            className={clsx(
              'mb-4 flex h-16 w-16 items-center justify-center rounded-2xl transition-colors',
              dragging ? 'bg-leaf-500 text-white' : 'bg-leaf-100 text-leaf-600 dark:bg-leaf-900/60 dark:text-leaf-300',
            )}
          >
            <Upload size={26} />
          </span>
          <p className="text-lg font-semibold">Drop a leaf photo here</p>
          <p className="mt-1.5 text-sm text-secondary">
            or <span className="text-leaf-600 font-medium">browse your files</span>
          </p>
          <p className="mt-3 text-xs text-muted">JPG, PNG or WEBP · up to 10 MB</p>

          <div className="mt-5 flex gap-2" onClick={(event) => event.stopPropagation()}>
            <Button variant="secondary" size="sm" icon={ImageIcon} onClick={() => inputRef.current?.click()}>
              Browse
            </Button>
            <Button
              variant="secondary"
              size="sm"
              icon={Camera}
              className="sm:hidden"
              onClick={() => cameraRef.current?.click()}
            >
              Camera
            </Button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="relative overflow-hidden rounded-2xl border border-[var(--border-subtle)] bg-black/5 dark:bg-black/30">
            <img src={preview} alt="Selected leaf" className="mx-auto max-h-[26rem] w-full object-contain" />
            {analysing && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/55 backdrop-blur-[2px]">
                <div className="absolute inset-x-0 h-0.5 bg-leaf-400 shadow-[0_0_18px_2px] shadow-leaf-400 animate-scan" />
                <Loader2 size={32} className="animate-spin text-white" />
                <p className="mt-3 text-sm font-medium text-white">Analysing leaf…</p>
                <p className="mt-1 text-xs text-white/70">Running the model and building heat maps</p>
              </div>
            )}
            {!analysing && (
              <button
                onClick={reset}
                className="absolute right-3 top-3 rounded-lg bg-black/60 p-2 text-white backdrop-blur transition hover:bg-black/80"
                aria-label="Remove image"
              >
                <X size={16} />
              </button>
            )}
          </div>

          {checking && (
            <div className="flex items-center gap-2 text-sm text-secondary">
              <Loader2 size={15} className="animate-spin" />
              Checking image quality…
            </div>
          )}

          {quality && (
            <div className={clsx('rounded-xl border p-3.5 text-sm', bandStyles[band])}>
              <div className="flex items-start gap-2.5">
                {band === 'good' ? (
                  <CheckCircle2 size={17} className="mt-0.5 shrink-0" />
                ) : (
                  <AlertTriangle size={17} className="mt-0.5 shrink-0" />
                )}
                <div className="min-w-0">
                  <p className="font-medium">
                    Image quality: {band} ({(quality.quality.score * 100).toFixed(0)}%)
                  </p>
                  <p className="mt-0.5 opacity-90">{quality.recommendation}</p>
                  {quality.quality.issues.length > 0 && (
                    <ul className="mt-2 space-y-1 text-xs opacity-90">
                      {quality.quality.issues.map((issue) => (
                        <li key={issue.code}>• {issue.message}</li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            </div>
          )}

          {error && (
            <p className="rounded-xl border border-clay-300 bg-clay-50 p-3 text-sm text-clay-800 dark:border-clay-800 dark:bg-clay-950/60 dark:text-clay-200">
              {error}
            </p>
          )}

          <div className="flex flex-col gap-2 sm:flex-row">
            <Button
              size="lg"
              className="flex-1"
              loading={analysing}
              disabled={disabled || checking}
              onClick={() => onAnalyse(file)}
            >
              {analysing ? 'Analysing…' : 'Analyse this leaf'}
            </Button>
            <Button variant="secondary" size="lg" onClick={reset} disabled={analysing}>
              Choose another
            </Button>
          </div>
        </div>
      )}

      {!preview && error && (
        <p className="mt-3 rounded-xl border border-clay-300 bg-clay-50 p-3 text-sm text-clay-800 dark:border-clay-800 dark:bg-clay-950/60 dark:text-clay-200">
          {error}
        </p>
      )}
    </div>
  );
}
