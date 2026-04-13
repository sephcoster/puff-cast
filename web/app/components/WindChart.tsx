"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface ChartPoint {
  time: string;       // ISO string
  timeLabel: string;  // formatted for display
  actual?: number;
  puffcast?: number;
  nws?: number;
  puffDir?: string;
  nwsDir?: string;
}

interface WindChartProps {
  data: ChartPoint[];
  station: string;
  nowLabel: string;
}

function CustomTooltip({ active, payload, label }: {
  active?: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  payload?: Array<any>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;

  // Find direction info from the data point
  const point = payload[0]?.payload as ChartPoint | undefined;

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm shadow-xl">
      <div className="text-slate-400 text-xs mb-1">{label}</div>
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2">
          <span
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-slate-300">{entry.name}:</span>
          <span className="font-bold text-slate-100">
            {Math.round(entry.value)} kt
          </span>
          {entry.name === "Puff Cast" && point?.puffDir && (
            <span className="text-slate-400 text-xs">{point.puffDir}</span>
          )}
          {entry.name === "NWS" && point?.nwsDir && (
            <span className="text-slate-400 text-xs">{point.nwsDir}</span>
          )}
        </div>
      ))}
    </div>
  );
}

export default function WindChart({ data, station, nowLabel }: WindChartProps) {
  if (!data.length) return null;

  // Find the max wind value for Y axis
  const maxWind = Math.max(
    ...data.map((d) =>
      Math.max(d.actual ?? 0, d.puffcast ?? 0, d.nws ?? 0)
    ),
  );
  const yMax = Math.ceil(maxWind / 5) * 5 + 5;

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={280}>
        <LineChart
          data={data}
          margin={{ top: 8, right: 16, left: 0, bottom: 0 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#1e293b"
            vertical={false}
          />
          <XAxis
            dataKey="timeLabel"
            tick={{ fontSize: 11, fill: "#64748b" }}
            tickLine={false}
            axisLine={{ stroke: "#334155" }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[0, yMax]}
            tick={{ fontSize: 11, fill: "#64748b" }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v: number) => `${v}`}
            label={{
              value: "kt",
              position: "insideTopLeft",
              offset: 8,
              style: { fontSize: 10, fill: "#475569" },
            }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 12, color: "#94a3b8" }}
          />

          {/* NOW line */}
          <ReferenceLine
            x={nowLabel}
            stroke="#38bdf8"
            strokeDasharray="4 4"
            strokeWidth={1}
            label={{
              value: "now",
              position: "top",
              style: { fontSize: 10, fill: "#38bdf8" },
            }}
          />

          {/* Actual observations — solid, bold */}
          <Line
            type="monotone"
            dataKey="actual"
            name="Actual"
            stroke="#f8fafc"
            strokeWidth={2.5}
            dot={false}
            connectNulls={false}
          />

          {/* Puff Cast prediction */}
          <Line
            type="monotone"
            dataKey="puffcast"
            name="Puff Cast"
            stroke="#22c55e"
            strokeWidth={2}
            strokeDasharray="6 3"
            dot={false}
            connectNulls={false}
          />

          {/* NWS HRRR */}
          <Line
            type="monotone"
            dataKey="nws"
            name="NWS"
            stroke="#f59e0b"
            strokeWidth={1.5}
            strokeDasharray="3 3"
            dot={false}
            connectNulls={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
