// frontend/src/pages/AspectAnalysis.jsx
import React, { useEffect, useMemo, useState } from "react";
import { Bar } from "react-chartjs-2";
import "chart.js/auto";
import ChartDataLabels from 'chartjs-plugin-datalabels';
import { getAspectSentimentSplit } from "../api";
import { useDate } from "../contexts/DateContext";

export default function AspectAnalysis() {
  const { start, end } = useDate();
  const [split, setSplit] = useState(null);
  const [splitAsPercent, setSplitAsPercent] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  // ---- load aspects × sentiment split ----
  useEffect(() => {
    if (!start || !end) return;
    (async () => {
      try {
        setLoading(true);
        setErr("");
        const s = await getAspectSentimentSplit(start, end, splitAsPercent);
        setSplit(s);
      } catch (e) {
        setErr("Failed to load aspect data");
        console.error("Failed to load aspect split", e);
      } finally {
        setLoading(false);
      }
    })();
  }, [start, end, splitAsPercent]);

  const stackedBarData = useMemo(() => {
    if (!split) return { labels: [], datasets: [] };
    const labels = split.labels || [];
    const series = splitAsPercent ? split.percent : split.counts;
    
    // Calculate total counts for each aspect
    const totals = labels.map((_, index) => {
      const pos = series.positive?.[index] || 0;
      const neu = series.neutral?.[index] || 0;
      const neg = series.negative?.[index] || 0;
      return pos + neu + neg;
    });
    
    return {
      labels: labels.map((t) => t.toUpperCase()),
      datasets: [
        { 
          label: splitAsPercent ? "% Positive" : "Positive", 
          data: series.positive || [], 
          backgroundColor: "#22c55e", 
          stack: "s",
          totals: totals
        },
        { 
          label: splitAsPercent ? "% Neutral"  : "Neutral",  
          data: series.neutral  || [], 
          backgroundColor: "#facc15", 
          stack: "s",
          totals: totals
        },
        { 
          label: splitAsPercent ? "% Negative" : "Negative", 
          data: series.negative || [], 
          backgroundColor: "#ef4444", 
          stack: "s",
          totals: totals
        },
      ],
    };
  }, [split, splitAsPercent]);

  const stackedBarOptions = {
    responsive: true,
    plugins: { 
      legend: { position: "top" },
      datalabels: {
        display: true,
        color: 'white',
        font: {
          weight: 'bold',
          size: 11
        },
        formatter: (value, context) => {
          const dataset = context.dataset;
          const totals = dataset.totals;
          const total = totals[context.dataIndex];
          const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
          return `${percentage}%`;
        }
      },
      tooltip: {
        callbacks: {
          afterLabel: function(context) {
            const dataset = context.dataset;
            const totals = dataset.totals;
            const total = totals[context.dataIndex];
            const value = context.parsed.y;
            const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
            return `Total: ${total} | ${percentage}%`;
          }
        }
      }
    },
    scales: {
      x: { stacked: true },
      y: {
        stacked: true,
        suggestedMax: splitAsPercent ? 100 : undefined,
        ticks: { callback: (v) => (splitAsPercent ? `${v}%` : v) },
      },
    },
  };

  return (
    <div className="space-y-4">
      {/* Error & Loading States */}
      {err && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 mb-4">
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            <span className="text-red-300 font-semibold text-sm">{err}</span>
          </div>
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-8">
          <div className="flex items-center space-x-3">
            <div className="w-6 h-6 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
            <span className="text-slate-300 font-semibold">Loading aspect analysis...</span>
          </div>
        </div>
      )}

      {!loading && split && (
        <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl p-4 border border-slate-600/30 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-xl flex items-center justify-center">
                <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <h2 className="text-lg font-bold text-white">Aspect × Sentiment Analysis</h2>
            </div>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={splitAsPercent}
                onChange={(e) => setSplitAsPercent(e.target.checked)}
                className="sr-only"
              />
              <div className={`relative w-12 h-6 rounded-full transition-colors ${
                splitAsPercent ? 'bg-gradient-to-r from-emerald-500 to-cyan-500' : 'bg-slate-600'
              }`}>
                <div className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform shadow-lg ${
                  splitAsPercent ? 'transform translate-x-6' : ''
                }`}></div>
              </div>
              <span className="text-xs font-semibold text-slate-300">Show as percentages</span>
            </label>
          </div>
          <div className="h-80">
            <Bar 
              data={stackedBarData} 
              options={{
                ...stackedBarOptions,
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  ...stackedBarOptions.plugins,
                  datalabels: {
                    ...stackedBarOptions.plugins.datalabels,
                    display: !splitAsPercent // Only show percentages inside bars when not in % mode
                  },
                  legend: {
                    ...stackedBarOptions.plugins.legend,
                    labels: {
                      color: '#e2e8f0',
                      font: {
                        size: 12,
                        weight: 'bold'
                      }
                    }
                  }
                },
                scales: {
                  ...stackedBarOptions.scales,
                  y: {
                    ...stackedBarOptions.scales.y,
                    grid: {
                      color: 'rgba(148, 163, 184, 0.1)'
                    },
                    ticks: {
                      color: '#94a3b8',
                      font: {
                        size: 11
                      }
                    }
                  },
                  x: {
                    ...stackedBarOptions.scales.x,
                    grid: {
                      display: false
                    },
                    ticks: {
                      color: '#94a3b8',
                      font: {
                        size: 11
                      }
                    }
                  }
                }
              }}
              plugins={[ChartDataLabels]}
            />
          </div>
        </div>
      )}
    </div>
  );
}
