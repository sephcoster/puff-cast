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
  valid_time: string;
  init_time: string;
}

interface StationForecast {
  name: string;
  leads: Record<string, LeadForecast>;
  current_kt?: number;
  current_time?: string;
}

interface Forecast {
  generated_at: string;
  hrrr_init: string | null;
  stations: Record<string, StationForecast>;
}

interface Verification {
  station: string;
  station_name: string;
  lead_hours: number;
  forecast_time: string;
  valid_time: string;
  predicted_kt: number;
  actual_kt: number;
  error_kt: number;
  abs_error_kt: number;
}

interface FunnelPrediction {
  predicted_kt: number;
  error_kt: number;
}

interface Funnel {
  station: string;
  station_name: string;
  valid_time: string;
  actual_kt: number;
  predictions: Record<string, FunnelPrediction>;
}

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

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/New_York",
  });
}

function formatDateTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
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

async function getForecast(): Promise<Forecast | null> {
  return readJson<Forecast | null>("latest.json", null);
}

async function getVerifications(): Promise<Verification[]> {
  return readJson<Verification[]>("verification.json", []);
}

async function getFunnels(): Promise<Funnel[]> {
  return readJson<Funnel[]>("funnels.json", []);
}

const ACCURACY_STATS = [
  { station: "Annapolis", ours: 1.4, nws: 4.4, pct: 70 },
  { station: "Cambridge", ours: 2.0, nws: 3.0, pct: 34 },
  { station: "Solomons", ours: 1.9, nws: 2.5, pct: 25 },
  { station: "Thomas Point", ours: 2.4, nws: 3.1, pct: 19 },
];

const LEAD_ORDER = [12, 6, 3, 1];

export default async function Home() {
  const [forecast, verifications, funnels] = await Promise.all([
    getForecast(),
    getVerifications(),
    getFunnels(),
  ]);

  return (
    <main className="max-w-3xl mx-auto px-4 py-8 space-y-6">
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
              {formatDateTime(forecast.generated_at)}
            </span>
          </span>
          {forecast.hrrr_init && (
            <span>
              HRRR init:{" "}
              <span className="text-slate-300">
                {formatTime(forecast.hrrr_init)}
              </span>
            </span>
          )}
          <span>Refreshed hourly</span>
        </div>
      )}

      {/* Forecast table */}
      {forecast ? (
        <div className="bg-slate-900 rounded-lg overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="bg-slate-800 text-xs uppercase tracking-wider text-slate-500">
                <th className="px-4 py-3 text-left">Station</th>
                <th className="px-3 py-3 text-center">Now</th>
                <th className="px-3 py-3 text-center">+1h</th>
                <th className="px-3 py-3 text-center">+3h</th>
                <th className="px-3 py-3 text-center">+6h</th>
                <th className="px-3 py-3 text-center">+12h</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(forecast.stations).map(([sid, data]) => (
                <tr
                  key={sid}
                  className="border-t border-slate-800 hover:bg-slate-800/50"
                >
                  <td className="px-4 py-3">
                    <div className="font-medium">{data.name}</div>
                    <div className="text-xs text-slate-500">{sid}</div>
                  </td>
                  <td className="px-3 py-3 text-center">
                    {data.current_kt != null ? (
                      <div>
                        <span className={windColor(data.current_kt)}>
                          {Math.round(data.current_kt)} kt
                        </span>
                        {data.current_time && (
                          <div className="text-xs text-slate-600">
                            {formatTime(data.current_time)}
                          </div>
                        )}
                      </div>
                    ) : (
                      <span className="text-slate-600">&mdash;</span>
                    )}
                  </td>
                  {["1", "3", "6", "12"].map((lead) => {
                    const ldata = data.leads[lead];
                    if (!ldata) {
                      return (
                        <td
                          key={lead}
                          className="px-3 py-3 text-center text-slate-600"
                        >
                          &mdash;
                        </td>
                      );
                    }
                    return (
                      <td key={lead} className="px-3 py-3 text-center">
                        <span
                          className={`text-lg font-bold ${windColor(ldata.wspd_kt)}`}
                        >
                          {Math.round(ldata.wspd_kt)} kt
                        </span>
                        <div className="text-xs text-slate-600">
                          {formatTime(ldata.valid_time)}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="bg-slate-900 rounded-lg p-8 text-center text-slate-500">
          Loading forecast...
        </div>
      )}

      {/* Wind speed legend */}
      <div className="flex flex-wrap gap-4 justify-center text-xs">
        {[
          ["slate-400", "<5 kt"],
          ["green-400", "5-10 kt"],
          ["blue-400", "10-15 kt"],
          ["amber-400", "15-20 kt"],
          ["red-400", "20+ kt"],
        ].map(([color, label]) => (
          <span key={label} className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${windDot(parseInt(label!))}`} />
            {label}
          </span>
        ))}
      </div>

      {/* Forecast Funnels */}
      {funnels.length > 0 && (
        <div className="bg-slate-900 rounded-lg p-4 space-y-3">
          <div>
            <h2 className="text-sky-400 font-semibold">
              Forecast Funnels
            </h2>
            <p className="text-xs text-slate-500">
              How did predictions converge as each hour approached?
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs uppercase tracking-wider text-slate-500">
                  <th className="px-3 py-2 text-left">Station</th>
                  <th className="px-2 py-2 text-center">Time</th>
                  {LEAD_ORDER.map((l) => (
                    <th key={l} className="px-2 py-2 text-center">
                      {l}h pred
                    </th>
                  ))}
                  <th className="px-2 py-2 text-center">Actual</th>
                </tr>
              </thead>
              <tbody>
                {funnels.slice(0, 20).map((f, i) => (
                  <tr
                    key={i}
                    className="border-t border-slate-800 hover:bg-slate-800/50"
                  >
                    <td className="px-3 py-2 whitespace-nowrap">
                      {f.station_name}
                    </td>
                    <td className="px-2 py-2 text-center text-slate-400 whitespace-nowrap">
                      {formatDateTime(f.valid_time)}
                    </td>
                    {LEAD_ORDER.map((lead) => {
                      const p = f.predictions[String(lead)];
                      if (!p) {
                        return (
                          <td
                            key={lead}
                            className="px-2 py-2 text-center text-slate-700"
                          >
                            &mdash;
                          </td>
                        );
                      }
                      const sign = p.error_kt > 0 ? "+" : "";
                      return (
                        <td key={lead} className="px-2 py-2 text-center">
                          <span className={windColor(p.predicted_kt)}>
                            {Math.round(p.predicted_kt)} kt
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
                        {Math.round(f.actual_kt)} kt
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Model accuracy */}
      <div className="bg-slate-900 rounded-lg p-4 space-y-3">
        <h2 className="text-sky-400 font-semibold">
          Model Accuracy (backtest, 12h lead)
        </h2>
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
      </footer>
    </main>
  );
}
