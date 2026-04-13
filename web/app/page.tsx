import { readFileSync, existsSync } from "fs";
import { join } from "path";
import WindChartWrapper from "./components/WindChartWrapper";

// In Vercel build, data is copied to public/data/ by the build script.
// Locally, read from ../docs/
const DOCS_DIR = existsSync(join(process.cwd(), "public", "data", "latest.json"))
  ? join(process.cwd(), "public", "data")
  : join(process.cwd(), "..", "docs");

interface LeadForecast {
  wspd_kt: number;
  wspd_ms: number;
  nws_kt: number | null;
  gust_kt?: number;
  nws_gust_kt?: number;
  dir_deg?: number;
  dir_cardinal?: string;
  dir_arrow?: string;
  nws_dir_deg?: number;
  nws_dir_cardinal?: string;
  valid_time: string;
  init_time: string;
}

interface StationForecast {
  name: string;
  leads: Record<string, LeadForecast>;
  current_kt?: number;
  current_time?: string;
  actuals?: Record<string, number>;
}

interface Forecast {
  generated_at: string;
  hrrr_init: string | null;
  stations: Record<string, StationForecast>;
}

interface FunnelPrediction {
  predicted_kt: number;
  error_kt: number;
  nws_kt?: number;
  nws_error_kt?: number;
}

interface BacktestFunnel {
  station: string;
  station_name: string;
  valid_time: string;
  actual_kt: number;
  predictions: Record<string, FunnelPrediction>;
}

interface UpcomingPrediction {
  predicted_kt: number;
  nws_kt: number | null;
  gust_kt?: number;
  nws_gust_kt?: number;
  dir_cardinal?: string;
  dir_arrow?: string;
  nws_dir_cardinal?: string;
  generated_at: string;
}

interface UpcomingFunnel {
  station: string;
  station_name: string;
  valid_time: string;
  predictions: Record<string, UpcomingPrediction>;
}

// -- helpers --------------------------------------------------------------

function windColor(kt: number): string {
  if (kt < 5) return "text-slate-400";
  if (kt < 10) return "text-green-400";
  if (kt < 15) return "text-blue-400";
  if (kt < 20) return "text-amber-400";
  return "text-red-400";
}

function windDot(kt: number): string {
  if (kt < 5) return "bg-slate-400";
  if (kt < 10) return "bg-green-400";
  if (kt < 15) return "bg-blue-400";
  if (kt < 20) return "bg-amber-400";
  return "bg-red-400";
}

function errorColor(absErr: number): string {
  if (absErr <= 1.5) return "text-green-400";
  if (absErr <= 3.0) return "text-amber-400";
  return "text-red-400";
}

function formatHour(iso: string): string {
  return new Date(iso).toLocaleTimeString("en-US", {
    hour: "numeric",
    timeZone: "America/New_York",
  });
}

function easternDateKey(iso: string | Date): string {
  // YYYY-MM-DD in America/New_York — for comparing dates without TZ drift
  const d = typeof iso === "string" ? new Date(iso) : iso;
  return d.toLocaleDateString("en-CA", { timeZone: "America/New_York" });
}

function formatDayHour(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const dKey = easternDateKey(d);
  const nowKey = easternDateKey(now);
  const tomorrow = new Date(now.getTime() + 86400_000);
  const tomorrowKey = easternDateKey(tomorrow);

  const hour = d.toLocaleTimeString("en-US", {
    hour: "numeric",
    timeZone: "America/New_York",
  });

  if (dKey === nowKey) return `Today ${hour}`;
  if (dKey === tomorrowKey) return `Tomorrow ${hour}`;
  return d.toLocaleString("en-US", {
    weekday: "short",
    hour: "numeric",
    timeZone: "America/New_York",
  });
}

function formatUpdated(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/New_York",
  });
}

function readJson<T>(filename: string, fallback: T): T {
  const path = join(DOCS_DIR, filename);
  if (!existsSync(path)) return fallback;
  try {
    return JSON.parse(readFileSync(path, "utf-8"));
  } catch {
    return fallback;
  }
}

interface VerificationRecord {
  station: string;
  station_name: string;
  lead_hours: number;
  error_kt: number;
  abs_error_kt: number;
  nws_kt?: number;
  nws_error_kt?: number;
}

const STATION_NAMES: Record<string, string> = {
  APAM2: "Annapolis",
  TPLM2: "Thomas Point",
  SLIM2: "Solomons",
};

function computeAccuracyStats(verifications: VerificationRecord[]) {
  const leads = [1, 3, 6, 12, 18, 24];
  const stations = ["APAM2", "TPLM2", "SLIM2"];

  // Per-station overall
  const overall: Array<{
    station: string;
    ours: number;
    nws: number | null;
    pct: number | null;
    n: number;
  }> = [];

  for (const sid of stations) {
    const sv = verifications.filter((v) => v.station === sid);
    if (sv.length === 0) continue;
    const ourMae =
      sv.reduce((s, v) => s + Math.abs(v.error_kt), 0) / sv.length;
    const nwsVals = sv.filter((v) => v.nws_error_kt != null);
    const nwsMae =
      nwsVals.length > 0
        ? nwsVals.reduce((s, v) => s + Math.abs(v.nws_error_kt!), 0) /
          nwsVals.length
        : null;
    const pct =
      nwsMae != null ? Math.round(((nwsMae - ourMae) / nwsMae) * 100) : null;
    overall.push({
      station: STATION_NAMES[sid] ?? sid,
      ours: Math.round(ourMae * 10) / 10,
      nws: nwsMae != null ? Math.round(nwsMae * 10) / 10 : null,
      pct,
      n: sv.length,
    });
  }

  // Per-station, per-lead
  const byLead: Array<{
    station: string;
    lead: number;
    ours: number;
    nws: number | null;
    winner: string;
    n: number;
  }> = [];

  for (const sid of stations) {
    for (const lead of leads) {
      const sv = verifications.filter(
        (v) => v.station === sid && v.lead_hours === lead,
      );
      if (sv.length < 2) continue;
      const ourMae =
        sv.reduce((s, v) => s + Math.abs(v.error_kt), 0) / sv.length;
      const nwsVals = sv.filter((v) => v.nws_error_kt != null);
      const nwsMae =
        nwsVals.length > 0
          ? nwsVals.reduce((s, v) => s + Math.abs(v.nws_error_kt!), 0) /
            nwsVals.length
          : null;
      byLead.push({
        station: STATION_NAMES[sid] ?? sid,
        lead,
        ours: Math.round(ourMae * 10) / 10,
        nws: nwsMae != null ? Math.round(nwsMae * 10) / 10 : null,
        winner:
          nwsMae == null ? "—" : ourMae < nwsMae ? "Puff Cast" : "NWS",
        n: sv.length,
      });
    }
  }

  return { overall, byLead, total: verifications.length };
}

const LEAD_COLUMNS = [24, 18, 12, 6, 3, 1];
const STATION_ORDER = ["APAM2", "TPLM2", "SLIM2"];

// -- page -----------------------------------------------------------------

export default async function Home() {
  const forecast = readJson<Forecast | null>("latest.json", null);
  const verifications = readJson<VerificationRecord[]>("verification.json", []);
  const backtestFunnels = readJson<BacktestFunnel[]>("funnels.json", []);
  const upcomingFunnels = readJson<UpcomingFunnel[]>(
    "upcoming_funnels.json",
    [],
  );

  // Group upcoming by station
  const upcomingByStation: Record<string, UpcomingFunnel[]> = {};
  for (const f of upcomingFunnels) {
    if (!upcomingByStation[f.station]) upcomingByStation[f.station] = [];
    upcomingByStation[f.station].push(f);
  }
  // Sort each station's upcoming by valid_time ascending
  for (const sid in upcomingByStation) {
    upcomingByStation[sid].sort(
      (a, b) =>
        new Date(a.valid_time).getTime() - new Date(b.valid_time).getTime(),
    );
  }

  // Group backtest funnels by station (most recent first)
  const backtestByStation: Record<string, BacktestFunnel[]> = {};
  for (const f of backtestFunnels) {
    if (!backtestByStation[f.station]) backtestByStation[f.station] = [];
    backtestByStation[f.station].push(f);
  }
  for (const sid in backtestByStation) {
    backtestByStation[sid].sort(
      (a, b) =>
        new Date(b.valid_time).getTime() - new Date(a.valid_time).getTime(),
    );
  }

  const stationIds = Object.keys(forecast?.stations ?? {});
  const orderedStationIds = [
    ...STATION_ORDER.filter((s) => stationIds.includes(s)),
    ...stationIds.filter((s) => !STATION_ORDER.includes(s)),
  ];

  return (
    <main className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-sky-400">Puff Cast</h1>
        <p className="text-slate-500 text-sm">
          ML-enhanced wind forecasts for Chesapeake Bay
        </p>
      </div>

      {/* Meta */}
      {forecast && (
        <div className="bg-slate-900 rounded-lg px-4 py-3 text-sm text-slate-400 flex flex-wrap gap-x-4">
          <span>
            Updated:{" "}
            <span className="text-slate-300">
              {formatUpdated(forecast.generated_at)}
            </span>
          </span>
          {forecast.hrrr_init && (
            <span>
              HRRR init:{" "}
              <span className="text-slate-300">
                {formatHour(forecast.hrrr_init)}
              </span>
            </span>
          )}
          <span>Refreshed hourly</span>
        </div>
      )}

      {/* Current conditions summary */}
      {forecast && (
        <div className="bg-slate-900 rounded-lg overflow-hidden">
          <div className="bg-slate-800 px-4 py-2 text-xs uppercase tracking-wider text-slate-500">
            Current conditions
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-slate-800">
            {orderedStationIds.map((sid) => {
              const s = forecast.stations[sid];
              return (
                <div key={sid} className="bg-slate-900 px-4 py-3">
                  <div className="text-xs text-slate-500">{s.name}</div>
                  {s.current_kt != null ? (
                    <div>
                      <span
                        className={`text-2xl font-bold ${windColor(s.current_kt)}`}
                      >
                        {Math.round(s.current_kt)} kt
                      </span>
                      {s.current_time && (
                        <span className="text-xs text-slate-600 ml-2">
                          {formatHour(s.current_time)}
                        </span>
                      )}
                    </div>
                  ) : (
                    <span className="text-slate-600">no data</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Next hour summary — one row per station */}
      {(() => {
        // For each station, find the earliest upcoming hour with at least one prediction
        const nextRowByStation: Record<
          string,
          { date: Date; hoursAway: number; funnel: UpcomingFunnel } | null
        > = {};
        const now = new Date();
        for (const sid of orderedStationIds) {
          const upcoming = upcomingByStation[sid] ?? [];
          const firstWithData = upcoming.find(
            (f) =>
              new Date(f.valid_time).getTime() > now.getTime() &&
              Object.keys(f.predictions).length > 0,
          );
          if (firstWithData) {
            const d = new Date(firstWithData.valid_time);
            nextRowByStation[sid] = {
              date: d,
              hoursAway: (d.getTime() - now.getTime()) / 3600_000,
              funnel: firstWithData,
            };
          } else {
            nextRowByStation[sid] = null;
          }
        }
        const anyNext = Object.values(nextRowByStation).some((v) => v);
        if (!anyNext) return null;

        return (
          <div className="bg-slate-900 rounded-lg overflow-hidden">
            <div className="bg-slate-800 px-4 py-2 text-xs uppercase tracking-wider text-slate-500">
              Next forecast per station
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs uppercase tracking-wider text-slate-500 bg-slate-900">
                    <th className="px-3 py-2 text-left">Station</th>
                    <th className="px-3 py-2 text-left">Next hour</th>
                    {LEAD_COLUMNS.map((l) => (
                      <th key={l} className="px-2 py-2 text-center">
                        <div>T-{l}h</div>
                        <div className="text-[9px] font-normal normal-case tracking-normal text-slate-600">
                          ours | nws
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {orderedStationIds.map((sid) => {
                    const name = forecast?.stations[sid]?.name ?? sid;
                    const entry = nextRowByStation[sid];
                    return (
                      <tr
                        key={sid}
                        className="border-t border-slate-800 hover:bg-slate-800/50"
                      >
                        <td className="px-3 py-2">
                          <div className="font-medium text-slate-300">
                            {name}
                          </div>
                          <div className="text-xs text-slate-600">{sid}</div>
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap">
                          {entry ? (
                            <>
                              <div className="font-medium text-slate-300">
                                {formatDayHour(entry.date.toISOString())}
                              </div>
                              <div className="text-xs text-slate-600">
                                in {Math.round(entry.hoursAway)}h
                              </div>
                            </>
                          ) : (
                            <span className="text-slate-600">—</span>
                          )}
                        </td>
                        {LEAD_COLUMNS.map((lead) => {
                          const p = entry?.funnel.predictions[String(lead)];
                          if (!p) {
                            return (
                              <td
                                key={lead}
                                className="px-2 py-2 text-center text-slate-700"
                              >
                                ·
                              </td>
                            );
                          }
                          return (
                            <td key={lead} className="px-2 py-1">
                              <div className="flex justify-center gap-3">
                                {/* Puff Cast prediction */}
                                <div className="text-center min-w-[3rem]">
                                  <div className="flex items-center justify-center gap-0.5">
                                    {p.dir_arrow && (
                                      <span className="text-slate-400">
                                        {p.dir_arrow}
                                      </span>
                                    )}
                                    <span
                                      className={`font-bold text-lg ${windColor(p.predicted_kt)}`}
                                    >
                                      {Math.round(p.predicted_kt)}
                                    </span>
                                    {p.gust_kt != null && p.gust_kt > p.predicted_kt + 1 && (
                                      <span className="text-xs text-slate-400 ml-0.5">
                                        G{Math.round(p.gust_kt)}
                                      </span>
                                    )}
                                  </div>
                                  {p.dir_cardinal && (
                                    <div className="text-[10px] text-slate-500">
                                      {p.dir_cardinal}
                                    </div>
                                  )}
                                </div>
                                {/* NWS prediction */}
                                {p.nws_kt != null && (
                                  <div className="text-center min-w-[2.5rem] border-l border-slate-700 pl-2">
                                    <div className="text-sm text-slate-500">
                                      {Math.round(p.nws_kt)}
                                      {p.nws_gust_kt != null && p.nws_gust_kt > p.nws_kt + 1 && (
                                        <span className="text-[10px] ml-0.5">
                                          G{Math.round(p.nws_gust_kt)}
                                        </span>
                                      )}
                                    </div>
                                    {p.nws_dir_cardinal && (
                                      <div className="text-[10px] text-slate-600">
                                        {p.nws_dir_cardinal}
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        );
      })()}

      {/* Upcoming funnels per station — expandable details */}
      {orderedStationIds.map((sid) => {
        const station = forecast?.stations[sid];
        const upcoming = upcomingByStation[sid] ?? [];
        const backtest = backtestByStation[sid] ?? [];
        const name = station?.name ?? sid;

        if (upcoming.length === 0 && backtest.length === 0) return null;

        // Build contiguous 24-hour range starting at the next top-of-hour
        const now = new Date();
        const startHour = new Date(now);
        startHour.setMinutes(0, 0, 0);
        startHour.setHours(startHour.getHours() + 1);

        // Map funnel entries by rounded-hour ISO so we can match them
        const funnelByHour = new Map<string, UpcomingFunnel>();
        for (const f of upcoming) {
          const fh = new Date(f.valid_time);
          fh.setMinutes(0, 0, 0);
          funnelByHour.set(fh.toISOString(), f);
        }

        const hours: {
          date: Date;
          hoursAway: number;
          funnel: UpcomingFunnel | undefined;
        }[] = [];
        for (let i = 0; i < 24; i++) {
          const d = new Date(startHour.getTime() + i * 3600_000);
          hours.push({
            date: d,
            hoursAway: (d.getTime() - now.getTime()) / 3600_000,
            funnel: funnelByHour.get(d.toISOString()),
          });
        }

        // Build chart data: past actuals + past/future predictions
        // Use a map keyed by epoch hour to merge all sources
        type ChartRow = {
          time: string;
          timeLabel: string;
          actual?: number;
          puffcast?: number;
          nws?: number;
          puffDir?: string;
          nwsDir?: string;
        };
        const chartMap = new Map<number, ChartRow>();

        function getOrCreate(isoOrDate: string | Date): ChartRow {
          const d = typeof isoOrDate === "string" ? new Date(isoOrDate) : isoOrDate;
          const epochH = Math.round(d.getTime() / 3600_000); // hour bucket
          if (!chartMap.has(epochH)) {
            chartMap.set(epochH, {
              time: d.toISOString(),
              timeLabel: formatHour(d.toISOString()),
            });
          }
          return chartMap.get(epochH)!;
        }

        // Past actuals (from station.actuals — last 48h of observations)
        const actuals = station?.actuals ?? {};
        for (const [t, kt] of Object.entries(actuals)) {
          getOrCreate(t).actual = kt;
        }

        // Past predictions from backtest funnels (use shortest lead = most refined)
        for (const bf of backtest.slice(0, 48)) {
          const sortedLeads = Object.keys(bf.predictions)
            .map(Number)
            .sort((a, b) => a - b);
          if (sortedLeads.length === 0) continue;
          const best = bf.predictions[String(sortedLeads[0])];
          const row = getOrCreate(bf.valid_time);
          row.puffcast = best.predicted_kt;
          row.actual = bf.actual_kt;
          if (best.nws_kt != null) {
            row.nws = best.nws_kt;
          }
        }

        // Future predictions (from upcoming funnels — use shortest lead)
        for (const h of hours) {
          if (!h.funnel) continue;
          const preds = h.funnel.predictions;
          const sortedLeads = Object.keys(preds)
            .map(Number)
            .sort((a, b) => a - b);
          if (sortedLeads.length === 0) continue;
          const best = preds[String(sortedLeads[0])];
          const row = getOrCreate(h.date);
          row.puffcast = best.predicted_kt;
          row.nws = best.nws_kt ?? undefined;
          row.puffDir = best.dir_cardinal;
          row.nwsDir = best.nws_dir_cardinal;
        }

        // Sort by time
        const chartData = [...chartMap.values()].sort(
          (a, b) => new Date(a.time).getTime() - new Date(b.time).getTime(),
        );

        // Find the "now" label for the reference line
        const nowForChart = new Date();
        nowForChart.setMinutes(0, 0, 0);
        const nowLabel = formatHour(nowForChart.toISOString());

        return (
          <details
            key={sid}
            className="bg-slate-900 rounded-lg overflow-hidden group"
          >
            <summary className="bg-slate-800 px-4 py-3 flex items-baseline justify-between cursor-pointer list-none marker:content-none">
              <div className="flex items-baseline gap-2">
                <span className="text-slate-400 text-xs group-open:rotate-90 transition-transform inline-block">
                  ▶
                </span>
                <h2 className="font-semibold text-sky-400">{name}</h2>
                <span className="text-xs text-slate-500">
                  full 24-hour detail
                </span>
              </div>
              <span className="text-xs text-slate-500">{sid}</span>
            </summary>
            {/* Wind speed timeline chart */}
            {chartData.length > 3 && (
              <div className="px-4 pt-4 pb-2 border-t border-slate-800">
                <div className="text-xs text-slate-500 mb-2">
                  Wind speed timeline — white: actual, green: Puff Cast, amber: NWS
                </div>
                <WindChartWrapper
                  data={chartData}
                  station={name}
                  nowLabel={nowLabel}
                />
              </div>
            )}

            <div className="px-4 py-2 text-xs text-slate-500 border-t border-slate-800">
              Each row is a future hour. Columns show predictions made that far
              ahead — cells fill in right‑to‑left as the hour approaches.
            </div>

            {/* Upcoming table — contiguous 24h window */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs uppercase tracking-wider text-slate-500 bg-slate-900">
                    <th className="px-3 py-2 text-left">Hour</th>
                    {LEAD_COLUMNS.map((l) => (
                      <th key={l} className="px-2 py-2 text-center">
                        <div>T-{l}h</div>
                        <div className="text-[9px] font-normal normal-case tracking-normal text-slate-600">
                          ours | nws
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {hours.map(({ date, hoursAway, funnel }, i) => (
                    <tr
                      key={i}
                      className="border-t border-slate-800 hover:bg-slate-800/50"
                    >
                      <td className="px-3 py-2 whitespace-nowrap">
                        <div className="font-medium text-slate-300">
                          {formatDayHour(date.toISOString())}
                        </div>
                        <div className="text-xs text-slate-600">
                          in {Math.round(hoursAway)}h
                        </div>
                      </td>
                      {LEAD_COLUMNS.map((lead) => {
                        const p = funnel?.predictions[String(lead)];
                        // A cell is "predictable" only when the lead is >= hoursAway.
                        // For shorter leads than hoursAway, we haven't had a chance yet.
                        const canPredict = lead >= hoursAway - 0.5;
                        if (!p) {
                          return (
                            <td
                              key={lead}
                              className={`px-2 py-2 text-center ${
                                canPredict ? "text-slate-700" : "text-slate-900"
                              }`}
                            >
                              {canPredict ? "·" : ""}
                            </td>
                          );
                        }
                        return (
                          <td key={lead} className="px-2 py-1">
                            <div className="flex justify-center gap-2">
                              <div className="text-center">
                                <div className="flex items-center justify-center gap-0.5">
                                  {p.dir_arrow && (
                                    <span className="text-slate-400 text-sm">
                                      {p.dir_arrow}
                                    </span>
                                  )}
                                  <span
                                    className={`font-bold ${windColor(p.predicted_kt)}`}
                                  >
                                    {Math.round(p.predicted_kt)}
                                  </span>
                                  {p.gust_kt != null && p.gust_kt > p.predicted_kt + 1 && (
                                    <span className="text-xs text-slate-400 ml-0.5">
                                      G{Math.round(p.gust_kt)}
                                    </span>
                                  )}
                                </div>
                                {p.dir_cardinal && (
                                  <div className="text-[10px] text-slate-500">
                                    {p.dir_cardinal}
                                  </div>
                                )}
                              </div>
                              {p.nws_kt != null && (
                                <div className="text-center border-l border-slate-700 pl-1.5">
                                  <div className="text-xs text-slate-500">
                                    {Math.round(p.nws_kt)}
                                    {p.nws_gust_kt != null && p.nws_gust_kt > p.nws_kt + 1 && (
                                      <span className="text-[10px] ml-0.5">
                                        G{Math.round(p.nws_gust_kt)}
                                      </span>
                                    )}
                                  </div>
                                  {p.nws_dir_cardinal && (
                                    <div className="text-[10px] text-slate-600">
                                      {p.nws_dir_cardinal}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Backtest table */}
            {backtest.length > 0 && (
              <>
                <div className="bg-slate-800 px-4 py-2 text-xs uppercase tracking-wider text-slate-500 border-t border-slate-700">
                  Past hours — what we predicted vs what actually happened
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-xs uppercase tracking-wider text-slate-500 bg-slate-900">
                        <th className="px-3 py-2 text-left">Hour</th>
                        {LEAD_COLUMNS.filter((l) => l <= 12).map((l) => (
                          <th key={l} className="px-2 py-2 text-center">
                            T-{l}h
                          </th>
                        ))}
                        <th className="px-2 py-2 text-center">Actual</th>
                      </tr>
                    </thead>
                    <tbody>
                      {backtest.slice(0, 12).map((f, i) => (
                        <tr
                          key={i}
                          className="border-t border-slate-800 hover:bg-slate-800/50"
                        >
                          <td className="px-3 py-2 whitespace-nowrap text-slate-400">
                            {formatDayHour(f.valid_time)}
                          </td>
                          {LEAD_COLUMNS.filter((l) => l <= 12).map((lead) => {
                            const p = f.predictions[String(lead)];
                            if (!p) {
                              return (
                                <td
                                  key={lead}
                                  className="px-2 py-2 text-center text-slate-700"
                                >
                                  &middot;
                                </td>
                              );
                            }
                            const sign = p.error_kt > 0 ? "+" : "";
                            return (
                              <td
                                key={lead}
                                className="px-2 py-2 text-center"
                              >
                                <span
                                  className={`font-medium ${windColor(p.predicted_kt)}`}
                                >
                                  {Math.round(p.predicted_kt)}
                                </span>
                                <div
                                  className={`text-xs ${errorColor(Math.abs(p.error_kt))}`}
                                >
                                  {sign}
                                  {p.error_kt.toFixed(1)}
                                </div>
                              </td>
                            );
                          })}
                          <td className="px-2 py-2 text-center">
                            <span
                              className={`font-bold ${windColor(f.actual_kt)}`}
                            >
                              {Math.round(f.actual_kt)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </details>
        );
      })}

      {/* Cell legend */}
      <div className="bg-slate-900 rounded-lg px-4 py-3 text-xs text-slate-400 space-y-2">
        <div className="flex items-center gap-3">
          <div className="flex flex-col items-center leading-none">
            <span className="font-bold text-base text-green-400">7</span>
            <span className="text-[10px] text-slate-600">10</span>
          </div>
          <div>
            <span className="font-bold text-slate-300">Large number</span> = our
            prediction in knots &nbsp;·&nbsp;{" "}
            <span className="text-slate-600">smaller number</span> = raw NWS
            HRRR
          </div>
        </div>
        <div className="flex flex-wrap gap-3 pt-1 border-t border-slate-800">
          <span className="text-slate-500 uppercase tracking-wider text-[10px]">
            Wind:
          </span>
          {[
            [1, "<5 kt"],
            [7, "5-10"],
            [12, "10-15"],
            [17, "15-20"],
            [25, "20+"],
          ].map(([kt, label]) => (
            <span key={label} className="flex items-center gap-1.5">
              <span
                className={`w-2 h-2 rounded-full ${windDot(kt as number)}`}
              />
              {label}
            </span>
          ))}
        </div>
      </div>

      {/* Live model accuracy */}
      {(() => {
        const stats = computeAccuracyStats(verifications);
        if (stats.total === 0) return null;
        return (
          <div className="bg-slate-900 rounded-lg p-4 space-y-4">
            <div>
              <h2 className="text-sky-400 font-semibold">
                Live Accuracy — {stats.total} verified predictions
              </h2>
              <p className="text-xs text-slate-500">
                Measured against actual NDBC observations as each forecast hour
                passes
              </p>
            </div>

            {/* Overall by station */}
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs uppercase tracking-wider text-slate-500">
                  <th className="px-3 py-2 text-left">Station</th>
                  <th className="px-2 py-2 text-center">Puff Cast MAE</th>
                  <th className="px-2 py-2 text-center">NWS MAE</th>
                  <th className="px-2 py-2 text-center">vs NWS</th>
                  <th className="px-2 py-2 text-center">n</th>
                </tr>
              </thead>
              <tbody>
                {stats.overall.map((s) => (
                  <tr
                    key={s.station}
                    className="border-t border-slate-800 hover:bg-slate-800/50"
                  >
                    <td className="px-3 py-2 font-medium">{s.station}</td>
                    <td className="px-2 py-2 text-center">{s.ours} kt</td>
                    <td className="px-2 py-2 text-center">
                      {s.nws != null ? `${s.nws} kt` : "—"}
                    </td>
                    <td
                      className={`px-2 py-2 text-center font-medium ${
                        s.pct != null && s.pct > 0
                          ? "text-green-400"
                          : s.pct != null && s.pct < 0
                            ? "text-red-400"
                            : "text-slate-500"
                      }`}
                    >
                      {s.pct != null
                        ? s.pct > 0
                          ? `+${s.pct}% better`
                          : `${s.pct}%`
                        : "—"}
                    </td>
                    <td className="px-2 py-2 text-center text-slate-500">
                      {s.n}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* By lead time */}
            <details className="group">
              <summary className="cursor-pointer text-xs text-slate-400 hover:text-slate-300 list-none flex items-center gap-1">
                <span className="group-open:rotate-90 transition-transform inline-block">
                  ▶
                </span>
                Accuracy by lead time
              </summary>
              <table className="w-full text-sm mt-2">
                <thead>
                  <tr className="text-xs uppercase tracking-wider text-slate-500">
                    <th className="px-3 py-2 text-left">Station</th>
                    <th className="px-2 py-2 text-center">Lead</th>
                    <th className="px-2 py-2 text-center">Ours</th>
                    <th className="px-2 py-2 text-center">NWS</th>
                    <th className="px-2 py-2 text-center">Winner</th>
                    <th className="px-2 py-2 text-center">n</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.byLead.map((s, i) => (
                    <tr
                      key={i}
                      className="border-t border-slate-800 hover:bg-slate-800/50"
                    >
                      <td className="px-3 py-1">{s.station}</td>
                      <td className="px-2 py-1 text-center">{s.lead}h</td>
                      <td className="px-2 py-1 text-center">{s.ours}</td>
                      <td className="px-2 py-1 text-center">
                        {s.nws ?? "—"}
                      </td>
                      <td
                        className={`px-2 py-1 text-center text-xs font-medium ${
                          s.winner === "Puff Cast"
                            ? "text-green-400"
                            : s.winner === "NWS"
                              ? "text-red-400"
                              : "text-slate-600"
                        }`}
                      >
                        {s.winner}
                      </td>
                      <td className="px-2 py-1 text-center text-slate-600">
                        {s.n}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          </div>
        );
      })()}

      {/* Footer */}
      <footer className="text-center text-xs text-slate-600 space-y-1 pb-8">
        <p>
          Ensemble of HRRR, GFS, ECMWF corrected with 27 local stations across
          the Bay.
        </p>
        <p>
          Each row is a future hour. Columns are lead times — when the
          prediction was made. Numbers show our prediction (large) and NWS
          (small) in knots.
        </p>
      </footer>
    </main>
  );
}
