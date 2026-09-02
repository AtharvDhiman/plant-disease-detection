import clsx from 'clsx';
import {
  Activity,
  BookOpen,
  FlaskConical,
  Code2,
  History,
  Leaf,
  LayoutDashboard,
  Menu,
  Moon,
  ScanLine,
  Sun,
  X,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link, NavLink, Outlet } from 'react-router-dom';

import { useTheme } from '../hooks/useTheme';
import { api } from '../services/api';

const NAV_ITEMS = [
  { to: '/', label: 'Analyse', icon: ScanLine, end: true },
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/benchmark', label: 'Model benchmark', icon: Activity },
  { to: '/research', label: 'Research', icon: FlaskConical },
  { to: '/library', label: 'Disease library', icon: BookOpen },
  { to: '/history', label: 'History', icon: History },
];

function BackendStatus() {
  const [status, setStatus] = useState({ state: 'checking' });

  useEffect(() => {
    let active = true;
    api
      .health()
      .then((data) => {
        if (!active) return;
        setStatus({
          state: data.model_ready ? 'ready' : 'no-model',
          model: data.production_model,
          device: data.device,
        });
      })
      .catch(() => active && setStatus({ state: 'offline' }));
    return () => {
      active = false;
    };
  }, []);

  const config = {
    checking: { dot: 'bg-neutral-400', text: 'Checking…' },
    ready: { dot: 'bg-leaf-500', text: `Model ready · ${status.device || ''}` },
    'no-model': { dot: 'bg-amber-500', text: 'No model exported' },
    offline: { dot: 'bg-clay-500', text: 'Backend offline' },
  }[status.state];

  return (
    <div className="flex items-center gap-2 text-xs text-muted" title={status.model || ''}>
      <span className={clsx('h-2 w-2 rounded-full', config.dot)} />
      <span className="hidden sm:inline truncate max-w-[14rem]">{config.text}</span>
    </div>
  );
}

export default function AppLayout() {
  const { theme, toggle } = useTheme();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Solid rather than translucent-with-backdrop-blur: a backdrop-filter on a
          sticky element forces a full-page compositing layer on every scroll frame,
          which is a measurable cost on low-end mobile hardware. */}
      <header className="sticky top-0 z-40 border-b border-[var(--border-subtle)] bg-[var(--surface-raised)]">
        <div className="mx-auto flex h-16 max-w-7xl items-center gap-3 px-4 sm:px-6">
          <Link to="/" className="flex items-center gap-2.5 shrink-0">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-leaf-600 text-white shadow-sm">
              <Leaf size={19} />
            </span>
            <span className="hidden sm:block leading-tight">
              <span className="block font-semibold tracking-tight">PlantGuard AI</span>
              <span className="block text-[11px] text-muted">CNN + CBAM disease detection</span>
            </span>
          </Link>

          <nav className="ml-4 hidden lg:flex items-center gap-1">
            {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-leaf-100 text-leaf-800 dark:bg-leaf-900/60 dark:text-leaf-200'
                      : 'text-[var(--text-secondary)] hover:bg-[var(--surface-sunken)] hover:text-[var(--text-primary)]',
                  )
                }
              >
                <Icon size={16} />
                {label}
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-3">
            <BackendStatus />
            <a
              href="/docs"
              target="_blank"
              rel="noreferrer"
              className="hidden md:inline-flex rounded-lg p-2 text-[var(--text-secondary)] hover:bg-[var(--surface-sunken)]"
              title="API documentation"
            >
              <Code2 size={17} />
            </a>
            <button
              onClick={toggle}
              className="rounded-lg p-2 text-[var(--text-secondary)] hover:bg-[var(--surface-sunken)]"
              aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            >
              {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
            </button>
            <button
              onClick={() => setMenuOpen((v) => !v)}
              className="lg:hidden rounded-lg p-2 text-[var(--text-secondary)] hover:bg-[var(--surface-sunken)]"
              aria-label="Toggle navigation menu"
              aria-expanded={menuOpen}
            >
              {menuOpen ? <X size={19} /> : <Menu size={19} />}
            </button>
          </div>
        </div>

        {menuOpen && (
          <nav className="lg:hidden border-t border-[var(--border-subtle)] bg-[var(--surface-raised)] px-4 py-2 sm:px-6">
            {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                onClick={() => setMenuOpen(false)}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 rounded-lg px-3 py-3 text-sm font-medium',
                    isActive
                      ? 'bg-leaf-100 text-leaf-800 dark:bg-leaf-900/60 dark:text-leaf-200'
                      : 'text-[var(--text-secondary)]',
                  )
                }
              >
                <Icon size={17} />
                {label}
              </NavLink>
            ))}
          </nav>
        )}
      </header>

      <main className="flex-1">
        <Outlet />
      </main>

      <footer className="border-t border-[var(--border-subtle)] bg-[var(--surface-raised)] mt-16">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
          <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
            <div className="max-w-md">
              <div className="flex items-center gap-2 font-semibold">
                <Leaf size={17} className="text-leaf-600" />
                PlantGuard AI
              </div>
              <p className="mt-2 text-sm text-secondary">
                Plant disease detection from leaf images using a convolutional neural network with
                a Convolutional Block Attention Module.
              </p>
            </div>
            <div className="text-sm">
              <p className="font-medium mb-2">Resources</p>
              <ul className="space-y-1.5 text-secondary">
                <li>
                  <a className="hover:text-leaf-600" href="/docs" target="_blank" rel="noreferrer">
                    API documentation (Swagger)
                  </a>
                </li>
                <li>
                  <a className="hover:text-leaf-600" href="/redoc" target="_blank" rel="noreferrer">
                    API reference (ReDoc)
                  </a>
                </li>
                <li>
                  <Link className="hover:text-leaf-600" to="/research">
                    Research dashboard
                  </Link>
                </li>
              </ul>
            </div>
          </div>
          <p className="mt-8 border-t border-[var(--border-subtle)] pt-5 text-xs text-muted">
            Predictions and disease information are provided for educational purposes only and are
            not a professional agricultural diagnosis. Confirm any suspected outbreak with a
            qualified plant pathologist or your local agricultural extension service before acting.
          </p>
        </div>
      </footer>
    </div>
  );
}
