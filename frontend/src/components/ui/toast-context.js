/**
 * Toast context and its consumer hook.
 *
 * Split out from `toast.jsx` because React Fast Refresh only works when a
 * module exports components exclusively; keeping the context and hook here lets
 * the provider file stay refreshable.
 */
import { createContext, useContext } from 'react';

export const ToastContext = createContext(null);

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) throw new Error('useToast must be used inside <ToastProvider>');
  return context;
}
