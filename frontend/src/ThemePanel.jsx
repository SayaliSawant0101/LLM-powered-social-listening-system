// frontend/src/ThemePanel.jsx
import React, { useState, useEffect } from "react";
import { fetchThemes } from "./api";
import { useDate } from "./contexts/DateContext";

const THEME_NAME_OVERRIDES = {
  "Hidden and Customer Support": "Hidden Charges and Customer Support",
};

const THEMES_PARQUET_OVERRIDE = import.meta?.env?.VITE_THEMES_PARQUET
  ? String(import.meta.env.VITE_THEMES_PARQUET).trim()
  : null;

// Use 15000 tweets by default (server will use this if max_rows is not provided)
const THEMES_MAX_ROWS = (() => {
  const raw = import.meta?.env?.VITE_THEMES_MAX_ROWS;
  if (!raw && raw !== 0) return 15000; // Default to 15000 if not set
  const parsed = Number.parseInt(raw, 10);
  return Number.isNaN(parsed) ? 15000 : parsed; // Default to 15000 if invalid
})();

const THEMES_DATASET_LABEL = THEMES_PARQUET_OVERRIDE
  ? THEMES_PARQUET_OVERRIDE.split(/[/\\]/).filter(Boolean).pop()
  : "tweets_stage2_aspects.parquet";

// ✅ New: cap clusters/themes to 8 always (auto + manual)
const MAX_THEMES = 8;

function formatThemeName(rawName) {
  if (!rawName) return "";

  const cleaned = String(rawName)
    .replace(/^"+|"+$/g, "")
    .trim()
    .replace(/\s+/g, " ");
  const normalized = cleaned.toLowerCase();

  const overrideEntry = Object.entries(THEME_NAME_OVERRIDES).find(
    ([key]) => key.toLowerCase() === normalized
  );

  return overrideEntry ? overrideEntry[1] : cleaned;
}

export default function ThemePanel() {
  const { start, end } = useDate();

  // ✅ Default becomes 8 instead of 12
  const [themeCount, setThemeCount] = useState(MAX_THEMES); // Active cluster count
  const [isAuto, setIsAuto] = useState(true); // Display mode: auto-detect label
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null); // {updated_at, themes: [...]}
  const [err, setErr] = useState("");

  useEffect(() => {
    if (start && end) {
      generateThemes();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [start, end]);

  async function generateThemes(options = {}) {
    if (!start || !end) return;

    const { auto = isAuto, count = themeCount } = options;

    // ✅ Key fix: always cap to MAX_THEMES
    // - Auto mode: don't force 12; let backend auto-detect if it supports it,
    //   but cap the value we send + display to 8.
    // - Manual mode: cap to 8.
    const requestedCountRaw = auto ? (Number.isFinite(count) ? count : MAX_THEMES) : (count || MAX_THEMES);
    const resolvedCount = Math.min(Number(requestedCountRaw) || MAX_THEMES, MAX_THEMES);

    setLoading(true);
    setErr("");

    try {
      const clusterDescriptor = auto
        ? `auto-detect (max ${MAX_THEMES}, requesting ${resolvedCount})`
        : `${resolvedCount} clusters`;

      console.log(
        `🚀 Generating themes dynamically for date range: ${start} to ${end} using ${clusterDescriptor}`
      );
      console.log(`⏱️  Optimized: Using parallel processing - should take 2-3 minutes!`);

      const payload = await fetchThemes({
        start: start || null,
        end: end || null,
        // ✅ Safety net: never send > 8
        n_clusters: resolvedCount,
        ...(THEMES_PARQUET_OVERRIDE ? { parquet: THEMES_PARQUET_OVERRIDE } : {}),
        ...(Number.isFinite(THEMES_MAX_ROWS) && THEMES_MAX_ROWS > 0
          ? { max_rows: THEMES_MAX_ROWS }
          : {}),
      });

      console.log("✅ Themes generated:", payload);
      console.log(
        "📊 Total tweets:",
        payload.themes?.reduce((sum, t) => sum + (t.tweet_count || 0), 0)
      );

      // ✅ Always cap what we display to 8
      const limitedThemes = Array.isArray(payload.themes)
        ? payload.themes.slice(0, resolvedCount)
        : [];

      setData({
        ...payload,
        themes: limitedThemes,
        total_cluster_count: payload.themes?.length ?? 0,
        source_row_count: payload.source_row_count ?? null,
      });
      setIsAuto(auto);
      setThemeCount(resolvedCount);
    } catch (e) {
      console.error("❌ Failed to generate themes:", e);
      let errorMessage = "Unknown error";

      if (e.response?.data) {
        const errorData = e.response.data;
        errorMessage = errorData.error || errorData.message || "Theme generation failed";

        if (errorData.details) {
          errorMessage += `: ${errorData.details}`;
        }

        if (errorData.hint) {
          errorMessage += ` (${errorData.hint})`;
        }
      } else if (e.message) {
        errorMessage = e.message;
      }

      setErr(`Failed to generate themes: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Theme Analysis</h2>
          <p className="text-slate-400 mt-1">
            {start && end ? `${start} to ${end}` : "Select date range"} • AI-powered theme
            discovery
          </p>

          {(Number.isFinite(THEMES_MAX_ROWS) && THEMES_MAX_ROWS > 0) || THEMES_PARQUET_OVERRIDE ? (
            <p className="text-slate-500 text-xs mt-1">
              {Number.isFinite(THEMES_MAX_ROWS) && THEMES_MAX_ROWS > 0
                ? `Limiting clustering to ${THEMES_MAX_ROWS.toLocaleString()} tweets`
                : null}
              {THEMES_PARQUET_OVERRIDE
                ? `${Number.isFinite(THEMES_MAX_ROWS) && THEMES_MAX_ROWS > 0 ? " • " : "Using "}Dataset: ${THEMES_DATASET_LABEL}`
                : null}
            </p>
          ) : null}
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-slate-300">Themes:</label>

            <select
              value={isAuto ? "auto" : String(themeCount)}
              onChange={(e) => {
                const val = e.target.value;
                const autoSelected = val === "auto";

                // ✅ Manual selection capped by options (6 or 8). Auto also capped by MAX_THEMES.
                const selectedCountRaw = autoSelected ? MAX_THEMES : parseInt(val, 10);
                const selectedCount = Math.min(
                  Number.isFinite(selectedCountRaw) ? selectedCountRaw : MAX_THEMES,
                  MAX_THEMES
                );

                setIsAuto(autoSelected);
                setThemeCount(selectedCount);

                if (start && end) {
                  generateThemes({ auto: autoSelected, count: selectedCount });
                }
              }}
              className="px-3 py-2 bg-slate-700/50 border border-slate-600/30 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
            >
              {/* ✅ Auto label updated + always max 8 */}
              <option value="auto">Auto-detect (Recommended, max {MAX_THEMES})</option>

              {/* ✅ Only allow <= 8 */}
              <option value="6">6 Themes</option>
              <option value="8">8 Themes</option>
            </select>
          </div>

          <button
            onClick={() => generateThemes()}
            disabled={loading}
            className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 disabled:from-slate-600 disabled:to-slate-700 text-white font-semibold rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl disabled:shadow-none disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>
                  {isAuto
                    ? `🔄 Auto-detecting themes (max ${MAX_THEMES})...`
                    : `🔄 Generating ${themeCount} themes...`}
                </span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
                <span>
                  {isAuto ? "🔄 Auto-detect Themes" : `🔄 Generate ${themeCount} Themes`}
                </span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error Message */}
      {err && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
          <div className="flex items-center space-x-2">
            <svg
              className="w-5 h-5 text-red-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span className="text-red-300 font-medium">Error</span>
          </div>
          <p className="text-red-200 mt-1">{err}</p>
        </div>
      )}

      {/* Results */}
      {data && data.themes && data.themes.length > 0 ? (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-white">{data.themes.length} Themes Identified</h3>
              <p className="text-sm text-slate-400 mt-1">
                Total:{" "}
                {data.themes
                  .reduce((sum, t) => sum + (t.tweet_count || 0), 0)
                  .toLocaleString()}{" "}
                tweets across all themes
              </p>

              {typeof data.source_row_count === "number" ? (
                <p className="text-xs text-slate-500 mt-1">
                  Source dataset: {data.source_row_count.toLocaleString()} tweets used for clustering
                </p>
              ) : null}

              {typeof data.total_cluster_count === "number" && data.total_cluster_count > data.themes.length ? (
                <p className="text-xs text-slate-500 mt-1">
                  Showing {data.themes.length} of {data.total_cluster_count} generated clusters to match your selection.
                </p>
              ) : null}
            </div>

            <p className="text-sm text-slate-400">
              Last updated: {data.updated_at ? new Date(data.updated_at).toLocaleString() : "—"}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.themes.map((t, index) => {
              // Calculate total tweets across all themes for the selected date range
              const totalTweets = data.themes.reduce(
                (sum, theme) => sum + (theme.tweet_count || 0),
                0
              );
              const percentage = totalTweets > 0 ? Math.round((t.tweet_count / totalTweets) * 100) : 0;
              const themeName = formatThemeName(t.name);

              return (
                <div
                  key={t.id ?? index}
                  className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl border border-slate-600/30 p-6 shadow-lg"
                >
                  {/* Theme Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div
                        className={`w-8 h-8 rounded-xl flex items-center justify-center text-white font-bold text-sm shadow-lg ${
                          index === 0
                            ? "bg-gradient-to-br from-yellow-500 to-orange-500"
                            : index === 1
                            ? "bg-gradient-to-br from-gray-500 to-slate-500"
                            : "bg-gradient-to-br from-emerald-500 to-cyan-500"
                        }`}
                      >
                        {index + 1}
                      </div>
                      <div>
                        <h4 className="text-lg font-bold text-white">
                          {themeName ? `"${themeName}"` : `Theme ${t.id ?? index + 1}`}
                        </h4>
                        <p className="text-sm text-slate-400 mt-1">
                          {t.tweet_count || 0} tweets • {isAuto ? "Auto-detected" : "Manual"} theme
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Theme Description */}
                  <p className="text-slate-300 text-sm leading-relaxed mb-4">
                    {t.summary || "No summary available for this theme."}
                  </p>

                  {/* Tweet Volume Section */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-slate-400">Tweet Volume</span>
                      <span className="text-sm text-slate-300 font-medium">
                        {t.tweet_count || 0} out of {totalTweets} tweets ({percentage}%)
                      </span>
                    </div>

                    {/* Progress bar */}
                    <div className="w-full bg-slate-700/50 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-emerald-500 to-cyan-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${percentage}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : data && data.themes && data.themes.length === 0 ? (
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-slate-700/50 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
          </div>
          <p className="text-slate-300 font-semibold">No themes found</p>
          <p className="text-sm text-slate-500 mt-1">
            Try adjusting your date range or check if there's enough data for theme generation.
          </p>
          <p className="text-xs text-slate-600 mt-2">Note: Theme generation may take 2-3 minutes to complete.</p>
        </div>
      ) : null}
    </div>
  );
}
