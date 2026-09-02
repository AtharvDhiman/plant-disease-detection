import { useCallback, useEffect, useState } from 'react';

const STORAGE_KEY = 'pdd-theme';

/** Read the stored preference, falling back to the OS setting. */
function initialTheme() {
  if (typeof window === 'undefined') return 'light';
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function useTheme() {
  const [theme, setTheme] = useState(initialTheme);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    window.localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  // Follow the OS only while the user has not made an explicit choice.
  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = (event) => {
      if (!window.localStorage.getItem(STORAGE_KEY)) {
        setTheme(event.matches ? 'dark' : 'light');
      }
    };
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, []);

  const toggle = useCallback(() => setTheme((t) => (t === 'dark' ? 'light' : 'dark')), []);
  return { theme, setTheme, toggle };
}
