import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Run an async loader on mount (and whenever `deps` change).
 *
 * Tracks a mounted flag so a slow request that resolves after navigation does
 * not set state on an unmounted component.
 */
export function useApi(loader, deps = [], { immediate = true } = {}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(immediate);
  const [error, setError] = useState(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await loader();
      if (mounted.current) setData(result);
      return result;
    } catch (err) {
      if (mounted.current) setError(err);
      return null;
    } finally {
      if (mounted.current) setLoading(false);
    }
    // The loader identity is intentionally not a dependency: callers pass an
    // inline arrow function, which would change on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    if (immediate) run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, immediate]);

  return { data, loading, error, reload: run, setData };
}
