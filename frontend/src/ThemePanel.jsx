// frontend/src/ThemePanel.jsx
import React, { useState, useEffect } from "react";
import { fetchThemes } from "./api";
import { useDate } from "./contexts/DateContext";

export default function ThemePanel() {
  const { start, end } = useDate();
  const [k, setK] = useState(6);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null); // {updated_at, themes: [...]}
  const [err, setErr] = useState("");


  async function run() {
    setLoading(true);
    setErr("");
    try {
      const payload = await fetchThemes({
        start: start || null,
        end: end || null,
        n_clusters: Number(k),
      });
      setData(payload);
    } catch (e) {
      setErr(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      {/* Modern Controls */}
      <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl p-4 border border-slate-600/30 shadow-xl">
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-gradient-to-r from-emerald-400 to-cyan-400 rounded-full"></div>
            <span className="font-semibold text-slate-200 text-sm">Theme Configuration</span>
        </div>
          
          <div className="flex items-center space-x-4">
        <div>
              <label className="block text-xs font-medium text-slate-400 mb-1"># Themes</label>
              <select
            value={k}
            onChange={(e) => setK(e.target.value)}
                className="px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-lg focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 transition-all w-24 text-white text-sm"
              >
                <option value={2}>2</option>
                <option value={3}>3</option>
                <option value={4}>4</option>
                <option value={5}>5</option>
                <option value={6}>6</option>
              </select>
              <div className="text-xs text-slate-500 mt-1">Top themes by tweet count</div>
        </div>
        <button
          onClick={run}
          disabled={loading}
              className="px-6 py-2 bg-gradient-to-r from-emerald-500 to-cyan-500 text-white rounded-lg font-semibold hover:from-emerald-600 hover:to-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-emerald-500/25 hover:scale-105 text-sm"
            >
            {loading ? (
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Generating themes...</span>
              </div>
            ) : (
              "Generate Themes"
            )}
        </button>
          </div>

          <div className="flex items-center space-x-3 ml-auto">
        {data?.updated_at && (
              <div className="text-xs text-slate-400 bg-slate-700/50 px-3 py-1 rounded-lg">
                Updated: {data.updated_at}
              </div>
            )}
            {err && (
              <div className="text-sm text-red-300 bg-red-500/10 px-3 py-1 rounded-lg border border-red-500/30">
                {err}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Results */}
      {!data?.themes?.length && !loading && !err && (
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-gradient-to-br from-slate-700/50 to-slate-600/50 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          </div>
          <p className="text-slate-300 font-semibold">No themes generated yet</p>
          <p className="text-sm text-slate-500 mt-1">Use the date range in the header and click "Generate Themes" to get started</p>
        </div>
      )}

      {data?.themes?.length ? (
        <div className="space-y-4">
          <div className="text-sm text-slate-300 bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 rounded-xl p-3 border border-emerald-500/20">
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="font-semibold text-sm">
                <strong className="text-emerald-400">{data.themes.filter(t => t.tweet_count > 0).length} themes found</strong> sorted by tweet volume (highest first)
              </span>
            </div>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {data.themes.filter(t => t.tweet_count > 0).map((t, index) => (
            <div key={t.id} className="group bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl p-4 border border-slate-600/30 shadow-xl hover:shadow-emerald-500/10 hover:border-emerald-400/30 transition-all duration-500">
              <div className="flex items-center space-x-3 mb-4">
                <div className={`w-8 h-8 rounded-xl flex items-center justify-center text-white font-bold text-sm shadow-lg ${
                  index === 0 ? 'bg-gradient-to-br from-yellow-500 to-amber-500' : 
                  index === 1 ? 'bg-gradient-to-br from-gray-400 to-gray-500' : 
                  index === 2 ? 'bg-gradient-to-br from-amber-600 to-orange-500' : 
                  'bg-gradient-to-br from-emerald-500 to-cyan-500'
                }`}>
                  {index + 1}
                </div>
                <h3 className="font-bold text-lg text-white leading-tight">
                  {t.name ? `"${String(t.name).replace(/^"+|"+$/g, "").trim()}"` : `Theme ${t.id}`}
                </h3>
              </div>

              <div className="bg-slate-700/30 rounded-xl p-3 border border-slate-600/20">
                <p className="text-slate-200 leading-relaxed text-sm">
                  {t.summary || "No summary available for this theme."}
                </p>
              </div>

              {/* Tweet count percentage */}
              <div className="mt-4 mb-4">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-medium">Tweet Volume</span>
                  <span className="font-bold text-white">
                    {t.tweet_count ?? 0} tweets
                    {data?.themes?.length && (
                      <span className="text-slate-400 ml-1">
                        ({Math.round(((t.tweet_count ?? 0) / data.themes.reduce((sum, theme) => sum + (theme.tweet_count ?? 0), 0)) * 100)}%)
                      </span>
                    )}
                  </span>
                </div>
                <div className="w-full bg-slate-700/50 rounded-full h-2 mt-2">
                  <div 
                    className="bg-gradient-to-r from-emerald-500 to-cyan-500 h-2 rounded-full transition-all duration-500 shadow-lg"
                    style={{ 
                      width: `${data?.themes?.length ? 
                        ((t.tweet_count ?? 0) / data.themes.reduce((sum, theme) => sum + (theme.tweet_count ?? 0), 0)) * 100 : 0}%` 
                    }}
                  ></div>
                </div>
              </div>

              {/* Progress bars for sentiment breakdown */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-medium">Sentiment Distribution</span>
                  <span>{t.tweet_count ?? 0} total</span>
                </div>
                <div className="flex h-2 bg-slate-700/50 rounded-full overflow-hidden">
                  <div 
                    className="bg-gradient-to-r from-green-500 to-emerald-500 h-full transition-all duration-500"
                    style={{ width: `${((t.positive ?? 0) / (t.tweet_count ?? 1)) * 100}%` }}
                  ></div>
                  <div 
                    className="bg-gradient-to-r from-yellow-500 to-amber-500 h-full transition-all duration-500"
                    style={{ width: `${((t.neutral ?? 0) / (t.tweet_count ?? 1)) * 100}%` }}
                  ></div>
                  <div 
                    className="bg-gradient-to-r from-red-500 to-pink-500 h-full transition-all duration-500"
                    style={{ width: `${((t.negative ?? 0) / (t.tweet_count ?? 1)) * 100}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs text-slate-500">
                  <span>Positive: {t.positive ?? 0}</span>
                  <span>Neutral: {t.neutral ?? 0}</span>
                  <span>Negative: {t.negative ?? 0}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
        </div>
      ) : null}
    </div>
  );
}
