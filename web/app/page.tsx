import { readFileSync, existsSync } from "fs";
import { join } from "path";

// In Vercel build, data is copied to public/data/ by the build script.
// Locally, read from ../docs/
const DOCS_DIR = existsSync(join(process.cwd(), "public", "data", "latest.json"))
  ? join(process.cwd(), "public", "data")
  : join(process.cwd(), "..", "docs");

interface LeadForecast {
  wspd_kt: number;
  wspd_ms: number;
  nws_kt: number | null;
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

const ACCURACY_STATS = [
  { station: "Annapolis", ours: 1.4, nws: 4.4, pct: 70 },
  { station: "Cambridge", ours: 2.0, nws: 3.0, pct: 34 },
  { station: "Solomons", ours: 1.9, nws: 2.5, pct: 25 },
  { station: "Thomas Point", ours: 2.4, nws: 3.1, pct: 19 },
];

const LEAD_COLUMNS = [24, 18, 12, 6, 3, 1];
const STATION_ORDER = ["APAM2", "TPLM2", "SLIM2", "CAMM2"];

// -- page -----------------------------------------------------------------

export default async function Home() {
  const forecast = readJson<Forecast | null>("latest.json", null);
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

      {/* Upcoming funnels per station */}
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

        return (
          <div key={sid} className="bg-slate-900 rounded-lg overflow-hidden">
            <div className="bg-slate-800 px-4 py-3 flex items-baseline justify-between">
              <div>
                <h2 className="font-semibold text-sky-400">{name}</h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Each row is a future hour. Columns show predictions made that
                  far ahead — cells fill in right‑to‑left as the hour
                  approaches.
                </p>
              </div>
              <span className="text-xs text-slate-500">{sid}</span>
            </div>

            {/* Upcoming table — contiguous 24h window */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs uppercase tracking-wider text-slate-500 bg-slate-900">
                    <th className="px-3 py-2 text-left">Hour</th>
                    {LEAD_COLUMNS.map((l) => (
                      <th key={l} className="px-2 py-2 text-center">
                        {l}h ahead
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
                          <td key={lead} className="px-2 py-2 text-center">
                            <span
                              className={`font-bold ${windColor(p.predicted_kt)}`}
                            >
                              {Math.round(p.predicted_kt)}
                            </span>
                            {p.nws_kt != null && (
                              <div className="text-xs text-slate-600">
                                {Math.round(p.nws_kt)}
                              </div>
                            )}
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
                            {l}h ahead
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
          </div>
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

      {/* Model accuracy */}
      <div className="bg-slate-900 rounded-lg p-4 space-y-3">
        <div>
          <h2 className="text-sky-400 font-semibold">Model Accuracy</h2>
          <p className="text-xs text-slate-500">
            Backtest at 12h lead — our predictions vs raw NWS HRRR
          </p>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs uppercase tracking-wider text-slate-500">
              <th className="px-3 py-2 text-left">Station</th>
              <th className="px-2 py-2 text-center">Puff Cast</th>
              <th className="px-2 py-2 text-center">Raw NWS</th>
              <th className="px-2 py-2 text-center">Improvement</th>
            </tr>
          </thead>
          <tbody>
            {ACCURACY_STATS.map((s) => (
              <tr
                key={s.station}
                className="border-t border-slate-800 hover:bg-slate-800/50"
              >
                <td className="px-3 py-2">{s.station}</td>
                <td className="px-2 py-2 text-center">{s.ours} kt</td>
                <td className="px-2 py-2 text-center">{s.nws} kt</td>
                <td className="px-2 py-2 text-center text-green-400 font-medium">
                  {s.pct}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

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
