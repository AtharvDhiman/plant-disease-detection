/**
 * Toast notifications.
 *
 * Kept in its own module rather than in the UI barrel file because exporting a
 * hook alongside components breaks React Fast Refresh for the whole module.
 */
import clsx from 'clsx';
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';

import { ToastContext } from './toast-context';

const TOAST_ICONS = {
  success: CheckCircle2,
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
};

const TOAST_TONES = {
  success:
    'border-leaf-300 bg-leaf-50 text-leaf-900 dark:bg-leaf-950 dark:text-leaf-100 dark:border-leaf-800',
  error:
    'border-clay-300 bg-clay-50 text-clay-900 dark:bg-clay-950 dark:text-clay-100 dark:border-clay-800',
  warning:
    'border-amber-300 bg-amber-50 text-amber-900 dark:bg-amber-950 dark:text-amber-100 dark:border-amber-800',
  info: 'border-[var(--border-strong)] bg-[var(--surface-raised)]',
};

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const dismiss = useCallback((id) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    (message, { type = 'info', duration = 5000, title } = {}) => {
      const id = Math.random().toString(36).slice(2);
      setToasts((current) => [...current, { id, message, type, title }]);
      if (duration) setTimeout(() => dismiss(id), duration);
      return id;
    },
    [dismiss],
  );

  const value = useMemo(() => ({ push, dismiss }), [push, dismiss]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        className="fixed bottom-4 right-4 z-50 flex w-[min(24rem,calc(100vw-2rem))] flex-col gap-2"
        role="region"
        aria-live="polite"
        aria-label="Notifications"
      >
        {toasts.map((toast) => {
          const Icon = TOAST_ICONS[toast.type];
          return (
            <div
              key={toast.id}
              className={clsx(
                'animate-fade-in-up flex items-start gap-3 rounded-xl border p-3.5 shadow-lg',
                TOAST_TONES[toast.type],
              )}
            >
              <Icon size={18} className="shrink-0 mt-0.5" />
              <div className="min-w-0 flex-1 text-sm">
                {toast.title && <p className="font-semibold">{toast.title}</p>}
                <p className={clsx(toast.title && 'mt-0.5 opacity-90')}>{toast.message}</p>
              </div>
              <button
                onClick={() => dismiss(toast.id)}
                className="shrink-0 opacity-60 hover:opacity-100"
                aria-label="Dismiss notification"
              >
                <X size={15} />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}
