// frontend/src/ThemePanel.jsx
import React, { useState, useEffect } from "react";
import { fetchThemes } from "./api";
import { useDate } from "./contexts/DateContext";

export default function ThemePanel() {
  const { start, end } = useDate();
  const [themeCount, setThemeCount] = useState(null); // null = auto-detect
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null); // {updated_at, themes: [...]}
  const [err, setErr] = useState("");

  async function run() {
    setLoading(true);
    setErr("");
    
    try {
      console.log(`Generating themes for date range: ${start} to ${end} with auto-detection`);
      
      const payload = await fetchThemes({
        start: start || null,
        end: end || null,
        n_clusters: themeCount, // Use selected theme count or auto-detect
      });
      
      console.log('Themes generated:', payload);
      setData(payload);
    } catch (e) {
      console.error('Failed to generate themes:', e);
      setErr(`Failed to generate themes: ${e.message}`);
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
            {start && end ? `${start} to ${end}` : 'All time'} • Auto-detected themes
          </p>
          </div>
          
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-slate-300">Themes:</label>
            <select
              value={themeCount || 'auto'}
              onChange={(e) => setThemeCount(e.target.value === 'auto' ? null : parseInt(e.target.value))}
              className="px-3 py-2 bg-slate-700/50 border border-slate-600/30 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
            >
              <option value="auto">Auto-detect (Recommended)</option>
              <option value="3">3 Themes</option>
              <option value="4">4 Themes</option>
              <option value="5">5 Themes</option>
              <option value="6">6 Themes</option>
              <option value="7">7 Themes</option>
              <option value="8">8 Themes</option>
            </select>
          </div>
          
          <button
            onClick={run}
            disabled={loading}
            className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 disabled:from-slate-600 disabled:to-slate-700 text-white font-semibold rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl disabled:shadow-none disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Generating themes... (this may take 2-3 minutes)</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>Generate Themes</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error Message */}
      {err && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
          <div className="flex items-center space-x-2">
            <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
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
            <h3 className="text-lg font-semibold text-white">
              Generated {data.themes.length} Themes
            </h3>
            <p className="text-sm text-slate-400">
              Last updated: {new Date(data.updated_at).toLocaleString()}
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.themes.map((t, index) => {
              // Debug: Log theme data to see what we're receiving
              console.log(`Theme ${index}:`, t);
              
              // Calculate total tweets across all themes for the selected date range
              const totalTweets = data.themes.reduce((sum, theme) => sum + theme.tweet_count, 0);
              const percentage = totalTweets > 0 ? Math.round((t.tweet_count / totalTweets) * 100) : 0;
              
              return (
                <div key={t.id} className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl border border-slate-600/30 p-6 shadow-lg">
                  {/* Theme Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                <div className={`w-8 h-8 rounded-xl flex items-center justify-center text-white font-bold text-sm shadow-lg ${
                        index === 0 ? 'bg-gradient-to-br from-yellow-500 to-orange-500' :
                        index === 1 ? 'bg-gradient-to-br from-gray-500 to-slate-500' :
                  'bg-gradient-to-br from-emerald-500 to-cyan-500'
                }`}>
                  {index + 1}
                </div>
                      <div>
                        <h4 className="text-lg font-bold text-white">
                  {t.name ? `"${String(t.name).replace(/^"+|"+$/g, "").trim()}"` : `Theme ${t.id}`}
                        </h4>
                        <p className="text-sm text-slate-400 mt-1">
                          {t.tweet_count} tweets • Auto-detected theme
                        </p>
                      </div>
                    </div>
              </div>

                  {/* Theme Description */}
                  <p className="text-slate-300 text-sm leading-relaxed mb-4">
                    {t.summary || 'No summary available for this theme.'}
                  </p>
                  
                  {/* Tweet Volume Section */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-slate-400">Tweet Volume</span>
                      <span className="text-sm text-slate-300 font-medium">
                        {t.tweet_count} out of {totalTweets} tweets ({percentage}%)
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
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <p className="text-slate-300 font-semibold">No themes found</p>
          <p className="text-sm text-slate-500 mt-1">
            Try adjusting your date range or check if there's enough data for theme generation.
          </p>
          <p className="text-xs text-slate-600 mt-2">
            Note: Theme generation may take 2-3 minutes to complete.
          </p>
        </div>
      ) : null}

    </div>
  );
}