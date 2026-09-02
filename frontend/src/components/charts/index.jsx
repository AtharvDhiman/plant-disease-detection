/**
 * Recharts wrappers with a single shared visual language.
 *
 * All theme-dependent colours come from CSS variables read at render time, so
 * charts follow the light/dark toggle without a second palette definition.
 */
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';

import { CHART_COLORS } from '../../utils/format';

const AXIS_PROPS = {
  stroke: 'var(--text-muted)',
  fontSize: 11,
  tickLine: false,
  axisLine: { stroke: 'var(--border-subtle)' },
};

function ChartTooltip({ active, payload, label, formatter }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-[var(--border-strong)] bg-[var(--surface-raised)] px-3 py-2 text-xs shadow-lg">
      {label !== undefined && <p className="mb-1 font-medium">{label}</p>}
      {payload.map((entry, index) => (
        <p key={index} className="flex items-center gap-2 text-[var(--text-secondary)]">
          <span className="h-2 w-2 rounded-full" style={{ background: entry.color }} />
          <span>{entry.name}:</span>
          <span className="font-medium tabular-nums text-[var(--text-primary)]">
            {formatter ? formatter(entry.value, entry.name) : entry.value}
          </span>
        </p>
      ))}
    </div>
  );
}

const grid = <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />;

/* ------------------------------------------------------------- bar charts */

export function HorizontalBarChart({
  data,
  dataKey,
  nameKey = 'label',
  height = 340,
  formatter,
  highlightKey = 'proposed',
  domain,
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 32, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" horizontal={false} />
        <XAxis type="number" domain={domain || [0, 'auto']} {...AXIS_PROPS} />
        <YAxis type="category" dataKey={nameKey} width={190} {...AXIS_PROPS} />
        <Tooltip content={<ChartTooltip formatter={formatter} />} cursor={{ fill: 'var(--surface-sunken)' }} />
        <Bar dataKey={dataKey} radius={[0, 5, 5, 0]} maxBarSize={22}>
          {data.map((entry, index) => (
            <Cell
              key={index}
              fill={entry[highlightKey] ? 'var(--color-clay-600)' : 'var(--color-leaf-600)'}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function GroupedBarChart({ data, series, nameKey = 'label', height = 340, formatter }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 8 }}>
        {grid}
        <XAxis dataKey={nameKey} {...AXIS_PROPS} interval={0} angle={-18} textAnchor="end" height={70} />
        <YAxis {...AXIS_PROPS} domain={[0, 1]} />
        <Tooltip content={<ChartTooltip formatter={formatter} />} cursor={{ fill: 'var(--surface-sunken)' }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((item, index) => (
          <Bar
            key={item.key}
            dataKey={item.key}
            name={item.label}
            fill={CHART_COLORS[index % CHART_COLORS.length]}
            radius={[4, 4, 0, 0]}
            maxBarSize={34}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function SimpleBarChart({ data, dataKey, nameKey, height = 280, color, formatter }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 8 }}>
        {grid}
        <XAxis dataKey={nameKey} {...AXIS_PROPS} />
        <YAxis {...AXIS_PROPS} allowDecimals={false} />
        <Tooltip content={<ChartTooltip formatter={formatter} />} cursor={{ fill: 'var(--surface-sunken)' }} />
        <Bar dataKey={dataKey} fill={color || 'var(--color-leaf-600)'} radius={[5, 5, 0, 0]} maxBarSize={48} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/* --------------------------------------------------------------- pie chart */

export function DonutChart({ data, dataKey = 'value', nameKey = 'name', height = 300 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={data}
          dataKey={dataKey}
          nameKey={nameKey}
          innerRadius="52%"
          outerRadius="80%"
          paddingAngle={2}
          stroke="var(--surface-raised)"
          strokeWidth={2}
        >
          {data.map((entry, index) => (
            <Cell key={index} fill={CHART_COLORS[index % CHART_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip content={<ChartTooltip />} />
        <Legend wrapperStyle={{ fontSize: 12 }} iconType="circle" />
      </PieChart>
    </ResponsiveContainer>
  );
}

/* -------------------------------------------------------------- line/area */

export function TrainingCurveChart({ data, series, height = 300, xKey = 'epoch', yDomain }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 8 }}>
        {grid}
        <XAxis dataKey={xKey} {...AXIS_PROPS} label={{ value: 'Epoch', position: 'insideBottom', offset: -4, fontSize: 11, fill: 'var(--text-muted)' }} />
        <YAxis {...AXIS_PROPS} domain={yDomain || ['auto', 'auto']} />
        <Tooltip content={<ChartTooltip formatter={(v) => Number(v).toFixed(4)} />} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((item, index) => (
          <Line
            key={item.key}
            type="monotone"
            dataKey={item.key}
            name={item.label}
            stroke={item.color || CHART_COLORS[index % CHART_COLORS.length]}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function TimelineChart({ data, height = 260 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 8 }}>
        <defs>
          <linearGradient id="timelineFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-leaf-500)" stopOpacity={0.42} />
            <stop offset="100%" stopColor="var(--color-leaf-500)" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        {grid}
        <XAxis dataKey="date" {...AXIS_PROPS} />
        <YAxis {...AXIS_PROPS} allowDecimals={false} />
        <Tooltip content={<ChartTooltip />} />
        <Area
          type="monotone"
          dataKey="count"
          name="Predictions"
          stroke="var(--color-leaf-600)"
          strokeWidth={2}
          fill="url(#timelineFill)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* ---------------------------------------------------------------- scatter */

export function TradeoffScatter({ data, xKey, yKey, xLabel, yLabel, height = 340, logX = false, formatter }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ left: 8, right: 24, top: 12, bottom: 24 }}>
        {grid}
        <XAxis
          type="number"
          dataKey={xKey}
          name={xLabel}
          scale={logX ? 'log' : 'auto'}
          domain={logX ? ['auto', 'auto'] : ['auto', 'auto']}
          {...AXIS_PROPS}
          label={{ value: xLabel, position: 'insideBottom', offset: -12, fontSize: 11, fill: 'var(--text-muted)' }}
        />
        <YAxis
          type="number"
          dataKey={yKey}
          name={yLabel}
          domain={['auto', 'auto']}
          {...AXIS_PROPS}
          label={{ value: yLabel, angle: -90, position: 'insideLeft', fontSize: 11, fill: 'var(--text-muted)' }}
        />
        <ZAxis range={[110, 110]} />
        <Tooltip
          cursor={{ strokeDasharray: '3 3' }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const point = payload[0].payload;
            return (
              <div className="rounded-lg border border-[var(--border-strong)] bg-[var(--surface-raised)] px-3 py-2 text-xs shadow-lg">
                <p className="font-medium">{point.label}</p>
                <p className="text-[var(--text-secondary)]">
                  {xLabel}: <span className="tabular-nums">{formatter?.x?.(point[xKey]) ?? point[xKey]}</span>
                </p>
                <p className="text-[var(--text-secondary)]">
                  {yLabel}: <span className="tabular-nums">{formatter?.y?.(point[yKey]) ?? point[yKey]}</span>
                </p>
              </div>
            );
          }}
        />
        <Scatter data={data}>
          {data.map((entry, index) => (
            <Cell key={index} fill={entry.proposed ? 'var(--color-clay-600)' : 'var(--color-leaf-600)'} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/* ----------------------------------------------------------------- radar */

export function MetricRadar({ data, series, height = 320 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RadarChart data={data} outerRadius="72%">
        <PolarGrid stroke="var(--border-subtle)" />
        <PolarAngleAxis dataKey="metric" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
        <PolarRadiusAxis domain={[0, 1]} tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
        <Tooltip content={<ChartTooltip formatter={(v) => Number(v).toFixed(4)} />} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((item, index) => (
          <Radar
            key={item.key}
            dataKey={item.key}
            name={item.label}
            stroke={CHART_COLORS[index % CHART_COLORS.length]}
            fill={CHART_COLORS[index % CHART_COLORS.length]}
            fillOpacity={0.18}
            strokeWidth={2}
          />
        ))}
      </RadarChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------- reliability plot */

export function ReliabilityChart({ bins, height = 300 }) {
  const data = (bins || [])
    .filter((bin) => bin.count > 0)
    .map((bin) => ({
      confidence: (bin.lower + bin.upper) / 2,
      accuracy: bin.accuracy,
      perfect: (bin.lower + bin.upper) / 2,
      count: bin.count,
    }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 16 }}>
        {grid}
        <XAxis
          dataKey="confidence"
          type="number"
          domain={[0, 1]}
          {...AXIS_PROPS}
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          label={{ value: 'Predicted confidence', position: 'insideBottom', offset: -8, fontSize: 11, fill: 'var(--text-muted)' }}
        />
        <YAxis
          domain={[0, 1]}
          {...AXIS_PROPS}
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
        />
        <Tooltip content={<ChartTooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line
          type="monotone"
          dataKey="accuracy"
          name="Empirical accuracy"
          stroke="var(--color-leaf-600)"
          strokeWidth={2.5}
          dot={{ r: 3 }}
        />
        <Line
          type="linear"
          dataKey="perfect"
          name="Perfect calibration"
          stroke="var(--color-clay-500)"
          strokeDasharray="5 4"
          strokeWidth={1.6}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
