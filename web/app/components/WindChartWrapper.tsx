"use client";

import dynamic from "next/dynamic";

const WindChart = dynamic(() => import("./WindChart"), {
  ssr: false,
  loading: () => (
    <div className="h-[280px] flex items-center justify-center text-slate-600 text-sm">
      Loading chart...
    </div>
  ),
});

interface ChartPoint {
  time: string;
  timeLabel: string;
  actual?: number;
  puffcast?: number;
  nws?: number;
  puffDir?: string;
  nwsDir?: string;
}

export default function WindChartWrapper({
  data,
  station,
  nowLabel,
}: {
  data: ChartPoint[];
  station: string;
  nowLabel: string;
}) {
  if (!data.length) return null;
  return <WindChart data={data} station={station} nowLabel={nowLabel} />;
}
