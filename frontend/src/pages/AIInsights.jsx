// frontend/src/pages/AIInsights.jsx
import React, { useState } from "react";
import { useDate } from "../contexts/DateContext";

export default function AIInsights() {
  const { start, end } = useDate();
  const [briefKeyword, setBriefKeyword] = useState("");

  // -------- NEW: Exec Summary + Structured Brief --------
  const [execData, setExecData] = useState(null); // {summary, stats, used_llm, ...}
  const [execLoading, setExecLoading] = useState(false);
  const [execErr, setExecErr] = useState("");

  const [briefData, setBriefData] = useState(null); // {executive_text, structured:{...}}
  const [briefLoading, setBriefLoading] = useState(false);
  const [briefErr, setBriefErr] = useState("");

  // ---- handlers for LLM endpoints ----
  async function runExecutiveSummary() {
    if (!start || !end) return;
    try {
      setExecLoading(true);
      setExecErr("");
      const q = new URLSearchParams({ start, end, sample_per_sentiment: String(250) }).toString();
      const r = await fetch(`http://127.0.0.1:8000/executive-summary?${q}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setExecData(j);
    } catch (e) {
      setExecErr(String(e));
      setExecData(null);
    } finally {
      setExecLoading(false);
    }
  }

  async function runStructuredBrief() {
    if (!start || !end) return;
    try {
      setBriefLoading(true);
      setBriefErr("");
      const q = new URLSearchParams({
        start,
        end,
        sample_size: String(80),
        ...(briefKeyword ? { keyword: briefKeyword } : {}),
      }).toString();
      const r = await fetch(`http://127.0.0.1:8000/structured-brief?${q}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setBriefData(j);
    } catch (e) {
      setBriefErr(String(e));
      setBriefData(null);
    } finally {
      setBriefLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      {/* Keyword Filter */}
      <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl border border-slate-600/30 shadow-xl p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-gradient-to-r from-emerald-400 to-cyan-400 rounded-full"></div>
            <span className="font-semibold text-slate-200 text-sm">Filter Options</span>
          </div>
          
          <div className="flex items-center space-x-3">
            <input
              type="text"
              placeholder="Filter by keyword..."
              value={briefKeyword}
              onChange={(e) => setBriefKeyword(e.target.value)}
              className="px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-lg focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400 transition-all w-64 text-white placeholder-slate-400 text-sm"
            />
          </div>
        </div>
      </div>

      {/* AI Insights Section */}
      <div className="space-y-4">
        {/* Executive Summary */}
        <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl p-4 border border-slate-600/30 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-xl flex items-center justify-center">
                <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h2 className="text-lg font-bold text-white">AI Executive Summary</h2>
            </div>
            <button
              onClick={runExecutiveSummary}
              disabled={execLoading || !start || !end}
              className="px-4 py-2 bg-gradient-to-r from-emerald-500 to-cyan-500 text-white rounded-lg font-semibold hover:from-emerald-600 hover:to-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-emerald-500/25 hover:scale-105 text-sm"
            >
              {execLoading ? (
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Generating...</span>
                </div>
              ) : (
                "Generate Summary"
              )}
            </button>
          </div>

          {execErr && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 mb-4">
              <div className="flex items-center space-x-2">
                <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <span className="text-red-300 font-semibold text-sm">Error: {execErr}</span>
              </div>
            </div>
          )}

          {execData && (
            <div className="space-y-3">
              <div className="flex items-center space-x-2 text-xs text-slate-400">
                <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                <span>{execData.used_llm ? "AI Model Active" : "Fallback Mode"}</span>
              </div>
              <div className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/20">
                <p className="text-slate-200 leading-relaxed whitespace-pre-wrap text-sm">
                  {execData.summary}
                </p>
              </div>
              {execData.stats && (
                <div className="flex items-center space-x-4 text-xs text-slate-400 bg-slate-700/30 rounded-lg p-2">
                  <span><strong>Stats:</strong></span>
                  <span>Negative: {execData.stats.sentiment?.counts?.negative ?? 0}</span>
                  <span>Neutral: {execData.stats.sentiment?.counts?.neutral ?? 0}</span>
                  <span>Positive: {execData.stats.sentiment?.counts?.positive ?? 0}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Structured Brief */}
        <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl p-4 border border-slate-600/30 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-xl flex items-center justify-center">
                <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h2 className="text-lg font-bold text-white">Structured Brief</h2>
            </div>
            <button
              onClick={runStructuredBrief}
              disabled={briefLoading || !start || !end}
              className="px-4 py-2 bg-gradient-to-r from-emerald-500 to-cyan-500 text-white rounded-lg font-semibold hover:from-emerald-600 hover:to-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-emerald-500/25 hover:scale-105 text-sm"
            >
              {briefLoading ? (
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Generating...</span>
                </div>
              ) : (
                "Generate Brief"
              )}
            </button>
          </div>

          {briefErr && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 mb-4">
              <div className="flex items-center space-x-2">
                <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <span className="text-red-300 font-semibold text-sm">Error: {briefErr}</span>
              </div>
            </div>
          )}

          {briefData && (
            <div className="space-y-4">
              {briefData.structured && Array.isArray(briefData.structured.executive_bullets) && briefData.structured.executive_bullets.length > 0 ? (
                <>
                  {/* Executive Bullets */}
                  <div>
                    <h3 className="text-sm font-semibold text-white mb-2 flex items-center space-x-2">
                      <span className="w-2 h-2 bg-emerald-400 rounded-full"></span>
                      <span>Executive Bullets</span>
                    </h3>
                    <div className="space-y-2">
                      {briefData.structured.executive_bullets.map((bullet, i) => (
                        <div key={i} className="flex items-start space-x-2 p-2 bg-slate-700/30 rounded-lg border border-slate-600/20">
                          <span className="w-5 h-5 bg-emerald-500 text-white text-xs rounded-full flex items-center justify-center font-bold mt-0.5">
                            {i + 1}
                          </span>
                          <span className="text-slate-200 text-sm">{bullet}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Themes */}
                  {Array.isArray(briefData.structured.themes) && briefData.structured.themes.length > 0 && (
                    <div>
                      <h3 className="text-sm font-semibold text-white mb-2 flex items-center space-x-2">
                        <span className="w-2 h-2 bg-cyan-400 rounded-full"></span>
                        <span>Key Themes</span>
                      </h3>
                      <div className="flex flex-wrap gap-2">
                        {briefData.structured.themes.map((theme, i) => (
                          <span key={i} className="px-2 py-1 bg-slate-700/30 text-slate-200 rounded-lg text-xs font-medium border border-slate-600/20">
                            {theme}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Risks */}
                  {Array.isArray(briefData.structured.risks) && briefData.structured.risks.length > 0 && (
                    <div>
                      <h3 className="text-sm font-semibold text-white mb-2 flex items-center space-x-2">
                        <span className="w-2 h-2 bg-red-400 rounded-full"></span>
                        <span>Risks</span>
                      </h3>
                      <div className="space-y-2">
                        {briefData.structured.risks.map((risk, i) => (
                          <div key={i} className="flex items-start space-x-2 p-2 bg-red-500/10 rounded-lg border border-red-500/20">
                            <svg className="w-4 h-4 text-red-400 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                            </svg>
                            <span className="text-slate-200 text-sm">{risk}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Opportunities */}
                  {Array.isArray(briefData.structured.opportunities) && briefData.structured.opportunities.length > 0 && (
                    <div>
                      <h3 className="text-sm font-semibold text-white mb-2 flex items-center space-x-2">
                        <span className="w-2 h-2 bg-green-400 rounded-full"></span>
                        <span>Opportunities</span>
                      </h3>
                      <div className="space-y-2">
                        {briefData.structured.opportunities.map((opportunity, i) => (
                          <div key={i} className="flex items-start space-x-2 p-2 bg-green-500/10 rounded-lg border border-green-500/20">
                            <svg className="w-4 h-4 text-green-400 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                            </svg>
                            <span className="text-slate-200 text-sm">{opportunity}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/20">
                  <p className="text-slate-200 leading-relaxed whitespace-pre-wrap text-sm">
                    {briefData.executive_text || "No content available"}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}