import clsx from 'clsx';
import { BookOpen, Filter, Leaf, Search, X } from 'lucide-react';
import { useMemo, useState } from 'react';

import DiseaseInfoPanel from '../components/DiseaseInfoPanel';
import {
  Button,
  Card,
  Chip,
  EmptyState,
  ErrorState,
  SkeletonCard,
} from '../components/ui';
import { useApi } from '../hooks/useApi';
import { api } from '../services/api';
import { CATEGORY_STYLES, SEVERITY_STYLES, number } from '../utils/format';

export default function Library() {
  const { data, loading, error, reload } = useApi(() => api.diseases(), []);
  const [search, setSearch] = useState('');
  const [plant, setPlant] = useState('');
  const [category, setCategory] = useState('');
  const [selected, setSelected] = useState(null);

  const items = useMemo(() => {
    if (!data?.items) return [];
    const needle = search.trim().toLowerCase();
    return data.items.filter((item) => {
      if (plant && item.plant !== plant) return false;
      if (category && item.category !== category) return false;
      if (!needle) return true;
      return (
        item.display_name.toLowerCase().includes(needle) ||
        item.common_name.toLowerCase().includes(needle) ||
        item.description.toLowerCase().includes(needle)
      );
    });
  }, [data, search, plant, category]);

  const { data: detail, loading: detailLoading } = useApi(
    () => (selected ? api.disease(selected) : Promise.resolve(null)),
    [selected],
  );

  if (loading) {
    return (
      <div className="mx-auto max-w-7xl space-y-6 px-4 py-10 sm:px-6">
        <SkeletonCard lines={2} />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <SkeletonCard key={index} lines={3} />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-10 sm:px-6">
        <ErrorState error={error} onRetry={reload} title="Could not load the disease library" />
      </div>
    );
  }

  const active = plant || category || search;

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-10 sm:px-6">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight">Disease library</h1>
        <p className="mt-2 max-w-3xl text-secondary">
          Reference information for every class the model can predict — {data.count} entries across{' '}
          {data.plants.length} plant species. Entries summarise widely documented plant-pathology
          guidance and deliberately contain no product names or spray schedules.
        </p>
      </header>

      <Card>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="relative lg:col-span-2">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search diseases…"
              className="w-full rounded-lg border border-[var(--border-strong)] bg-[var(--surface-raised)] py-2.5 pl-9 pr-3 text-sm"
            />
          </div>
          <select
            value={plant}
            onChange={(event) => setPlant(event.target.value)}
            className="rounded-lg border border-[var(--border-strong)] bg-[var(--surface-raised)] px-3 py-2.5 text-sm"
          >
            <option value="">All plants</option>
            {data.plants.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <select
            value={category}
            onChange={(event) => setCategory(event.target.value)}
            className="rounded-lg border border-[var(--border-strong)] bg-[var(--surface-raised)] px-3 py-2.5 text-sm"
          >
            <option value="">All categories</option>
            {data.categories.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
        {active && (
          <div className="mt-3 flex items-center gap-2">
            <span className="text-sm text-secondary">
              {items.length} of {data.count} entries
            </span>
            <button
              onClick={() => {
                setSearch('');
                setPlant('');
                setCategory('');
              }}
              className="inline-flex items-center gap-1 text-sm text-leaf-600 hover:underline"
            >
              <X size={13} />
              Clear filters
            </button>
          </div>
        )}
      </Card>

      {selected && (
        <div className="space-y-3">
          <Button variant="secondary" size="sm" icon={X} onClick={() => setSelected(null)}>
            Close detail
          </Button>
          {detailLoading ? <SkeletonCard lines={8} /> : <DiseaseInfoPanel info={detail} />}
        </div>
      )}

      {items.length === 0 ? (
        <Card>
          <EmptyState icon={Filter} title="No entries match those filters" />
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((item) => (
            <button
              key={item.class_name}
              onClick={() => {
                setSelected(item.class_name);
                window.scrollTo({ top: 260, behavior: 'smooth' });
              }}
              className={clsx(
                'surface-card p-5 text-left transition-shadow hover:shadow-lg',
                selected === item.class_name && 'ring-2 ring-leaf-500',
              )}
            >
              <div className="mb-3 flex items-start justify-between gap-2">
                <span
                  className={clsx(
                    'flex h-9 w-9 shrink-0 items-center justify-center rounded-lg',
                    item.is_healthy
                      ? 'bg-leaf-100 text-leaf-700 dark:bg-leaf-900/60 dark:text-leaf-300'
                      : 'bg-clay-100 text-clay-700 dark:bg-clay-900/60 dark:text-clay-300',
                  )}
                >
                  {item.is_healthy ? <Leaf size={17} /> : <BookOpen size={17} />}
                </span>
                <div className="flex flex-wrap justify-end gap-1.5">
                  <Chip className={clsx('border-transparent', CATEGORY_STYLES[item.category])}>
                    {item.category}
                  </Chip>
                </div>
              </div>

              <h3 className="font-semibold leading-tight">{item.common_name}</h3>
              <p className="mt-0.5 text-sm text-muted">{item.plant}</p>
              <p className="mt-2.5 line-clamp-3 text-sm leading-relaxed text-secondary">
                {item.description}
              </p>

              <div className="mt-3.5 flex items-center justify-between gap-2 border-t border-[var(--border-subtle)] pt-3">
                <Chip className={SEVERITY_STYLES[item.severity]}>
                  {item.severity === 'none' ? 'healthy' : item.severity}
                </Chip>
                <span className="text-xs text-muted">
                  {number(item.training_images)} training images
                </span>
              </div>
            </button>
          ))}
        </div>
      )}

      {data.metadata?.editorial_policy && (
        <Card className="bg-[var(--surface-sunken)]">
          <h3 className="font-semibold">Editorial policy</h3>
          <p className="mt-2 text-sm leading-relaxed text-secondary">
            {data.metadata.editorial_policy}
          </p>
          <p className="mt-3 text-sm leading-relaxed text-secondary">{data.metadata.disclaimer}</p>
          {data.metadata.sources?.length > 0 && (
            <p className="mt-3 text-xs text-muted">
              <span className="font-medium">Information drawn from: </span>
              {data.metadata.sources.join('; ')}
            </p>
          )}
        </Card>
      )}
    </div>
  );
}
