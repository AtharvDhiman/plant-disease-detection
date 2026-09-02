import clsx from 'clsx';
import {
  AlertOctagon,
  BookOpen,
  Bug,
  CloudRain,
  Leaf,
  ShieldCheck,
  Stethoscope,
  Wrench,
} from 'lucide-react';

import { CATEGORY_STYLES, SEVERITY_STYLES } from '../utils/format';
import { Card, CardHeader, Chip } from './ui';

const SECTIONS = [
  { key: 'symptoms', title: 'Symptoms to look for', icon: Stethoscope },
  { key: 'causes', title: 'How it spreads', icon: Bug },
  { key: 'favourable_conditions', title: 'Conditions that favour it', icon: CloudRain },
  { key: 'prevention', title: 'Prevention', icon: ShieldCheck },
  { key: 'management', title: 'General management', icon: Wrench },
];

export default function DiseaseInfoPanel({ info }) {
  if (!info) return null;

  const severity = info.severity || 'unknown';
  const category = info.category || 'unknown';

  return (
    <Card>
      <CardHeader
        title={info.common_name}
        subtitle={`${info.plant}${info.pathogen ? ` · ${info.pathogen}` : ''}`}
        icon={info.is_healthy ? Leaf : BookOpen}
        action={
          <div className="flex flex-wrap justify-end gap-2">
            <Chip className={clsx('border-transparent', CATEGORY_STYLES[category])}>{category}</Chip>
            {severity !== 'unknown' && (
              <Chip className={SEVERITY_STYLES[severity]}>
                {severity === 'none' ? 'no disease' : `${severity} severity`}
              </Chip>
            )}
          </div>
        }
      />

      <p className="text-sm leading-relaxed text-secondary">{info.description}</p>

      {severity === 'critical' && (
        <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-clay-300 bg-clay-50 p-3.5 text-sm text-clay-900 dark:border-clay-800 dark:bg-clay-950/60 dark:text-clay-100">
          <AlertOctagon size={17} className="mt-0.5 shrink-0" />
          <p>
            This condition is classed as <strong>critical</strong>: it can destroy a crop, or it has
            no cure once established. Act promptly and involve your local plant-health authority.
          </p>
        </div>
      )}

      <div className="mt-5 grid gap-5 sm:grid-cols-2">
        {SECTIONS.map(({ key, title, icon: Icon }) => {
          const items = info[key];
          if (!items?.length) return null;
          return (
            <div key={key}>
              <h4 className="mb-2 flex items-center gap-2 text-sm font-semibold">
                <Icon size={15} className="text-leaf-600" />
                {title}
              </h4>
              <ul className="space-y-1.5 text-sm text-secondary">
                {items.map((item, index) => (
                  <li key={index} className="flex gap-2 leading-relaxed">
                    <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-leaf-500" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>

      {info.disclaimer && (
        <p className="mt-6 rounded-lg border border-[var(--border-subtle)] bg-[var(--surface-sunken)] p-3 text-xs leading-relaxed text-muted">
          {info.disclaimer}
        </p>
      )}

      {info.sources?.length > 0 && (
        <div className="mt-3 text-xs text-muted">
          <span className="font-medium">Information drawn from: </span>
          {info.sources.join('; ')}
        </div>
      )}
    </Card>
  );
}
