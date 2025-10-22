// frontend/src/pages/Dashboard.jsx
import React, { useEffect, useMemo, useState } from "react";
import { Line } from "react-chartjs-2";
import "chart.js/auto";
import { getSummary, getTrend, getAspectSentimentSplit } from "../api";
import { useDate } from "../contexts/DateContext";

// --- helpers ---
function iso(x) {
  if (!x) return "";
  const d = new Date(x);
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

export default function Dashboard() {
  const { start, end, meta, setStart, setEnd } = useDate();
  const [summary, setSummary] = useState(null);
  const [trend, setTrend] = useState([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [timePeriod, setTimePeriod] = useState("daily");
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMoreData, setHasMoreData] = useState(true);
  const [currentOffset, setCurrentOffset] = useState(0);
  const [aspectSplitModal, setAspectSplitModal] = useState({ isOpen: false, sentiment: null, data: null });
  const [loadingAspectSplit, setLoadingAspectSplit] = useState(false);
  const [selectedDateModal, setSelectedDateModal] = useState({ isOpen: false, date: null, data: null });
  const [loadingDateAspects, setLoadingDateAspects] = useState(false);

  // ---- load sentiment summary + trend ----
  useEffect(() => {
    if (!start || !end) return;
    (async () => {
      try {
        setLoading(true);
        setErr("");
        setCurrentOffset(0);
        setHasMoreData(true);
        const [s, t] = await Promise.all([
          getSummary(start, end), 
          getTrend(start, end, timePeriod, 0, 50)
        ]);
        setSummary(s);
        setTrend(t?.trend || []);
        setHasMoreData((t?.trend || []).length === 50);
      } catch (error) {
        console.error('Failed to load data:', error);
        setErr("Failed to load data. Is the API running on :8000?");
      } finally {
        setLoading(false);
      }
    })();
  }, [start, end, timePeriod]);

  // ---- load more trend data ----
  const loadMoreData = async () => {
    if (loadingMore || !hasMoreData) return;
    
    try {
      setLoadingMore(true);
      const newOffset = currentOffset + 50;
      const t = await getTrend(start, end, timePeriod, newOffset, 50);
      const newData = t?.trend || [];
      
      if (newData.length > 0) {
        setTrend(prev => [...prev, ...newData]);
        setCurrentOffset(newOffset);
        setHasMoreData(newData.length === 50);
      } else {
        setHasMoreData(false);
      }
    } catch (error) {
      console.error("Failed to load more data:", error);
    } finally {
      setLoadingMore(false);
    }
  };

  // ---- load aspect split for selected date ----
  const loadDateAspects = async (selectedDate) => {
    try {
      setLoadingDateAspects(true);
      
      // Find the actual date value from trend data based on the formatted label
      const actualDate = trend.find(item => {
        const formattedDate = formatDate(item.date);
        return formattedDate === selectedDate;
      })?.date;
      
      if (!actualDate) {
        console.error('Could not find actual date for:', selectedDate);
        setErr("Could not find data for selected date");
        return;
      }
      
      // Use a small date range around the selected date to ensure we get data
      const dateObj = new Date(actualDate);
      const nextDay = new Date(dateObj);
      nextDay.setDate(dateObj.getDate() + 1);
      
      const startDate = actualDate;
      const endDate = nextDay.toISOString().split('T')[0];
      
      // Get aspect sentiment split for the date range
      const data = await getAspectSentimentSplit(startDate, endDate, true, true);
      
      // Transform the data to show three-level hierarchy: Total -> Aspect -> Sentiment
      const transformedData = {
        totalTweets: 0,
        aspectBreakdown: [],
        sentimentBreakdown: {
          positive: { total: 0, aspects: [] },
          neutral: { total: 0, aspects: [] },
          negative: { total: 0, aspects: [] }
        }
      };
      
      // Handle the API response structure: {labels: [], counts: {positive: [], negative: [], neutral: []}, percent: {positive: [], negative: [], neutral: []}}
      if (data?.labels && data?.counts && data?.percent) {
        // Calculate total tweets first
        const grandTotal = data.labels.reduce((sum, _, idx) => {
          return sum + (data.counts.positive[idx] || 0) + (data.counts.negative[idx] || 0) + (data.counts.neutral[idx] || 0);
        }, 0);
        
        // Calculate sentiment totals
        const positiveTotal = data.labels.reduce((sum, _, idx) => sum + (data.counts.positive[idx] || 0), 0);
        const neutralTotal = data.labels.reduce((sum, _, idx) => sum + (data.counts.neutral[idx] || 0), 0);
        const negativeTotal = data.labels.reduce((sum, _, idx) => sum + (data.counts.negative[idx] || 0), 0);
        
        // Transform data by aspect (overall level)
        transformedData.aspectBreakdown = data.labels.map((label, index) => {
          const positiveCount = data.counts.positive[index] || 0;
          const negativeCount = data.counts.negative[index] || 0;
          const neutralCount = data.counts.neutral[index] || 0;
          const totalCount = positiveCount + negativeCount + neutralCount;
          
          return {
            aspect: label,
            count: totalCount,
            percentage: grandTotal > 0 ? (totalCount / grandTotal) * 100 : 0
          };
        }).filter(item => item.count > 0); // Only show aspects with data
        
        // Transform data by sentiment with aspect breakdown
        transformedData.sentimentBreakdown.positive = {
          total: positiveTotal,
          aspects: data.labels.map((label, index) => ({
            aspect: label,
            count: data.counts.positive[index] || 0,
            percentage: positiveTotal > 0 ? ((data.counts.positive[index] || 0) / positiveTotal) * 100 : 0
          })).filter(item => item.count > 0)
        };
        
        transformedData.sentimentBreakdown.neutral = {
          total: neutralTotal,
          aspects: data.labels.map((label, index) => ({
            aspect: label,
            count: data.counts.neutral[index] || 0,
            percentage: neutralTotal > 0 ? ((data.counts.neutral[index] || 0) / neutralTotal) * 100 : 0
          })).filter(item => item.count > 0)
        };
        
        transformedData.sentimentBreakdown.negative = {
          total: negativeTotal,
          aspects: data.labels.map((label, index) => ({
            aspect: label,
            count: data.counts.negative[index] || 0,
            percentage: negativeTotal > 0 ? ((data.counts.negative[index] || 0) / negativeTotal) * 100 : 0
          })).filter(item => item.count > 0)
        };
        
        // Set total tweets
        transformedData.totalTweets = grandTotal;
      }
      
      setSelectedDateModal({
        isOpen: true,
        date: actualDate,
        formattedDate: selectedDate,
        data: transformedData
      });
    } catch (error) {
      console.error("Failed to load date aspects:", error);
      console.error("Error details:", error.response?.data || error.message);
      setErr(`Failed to load aspect breakdown for selected date: ${error.message}`);
    } finally {
      setLoadingDateAspects(false);
    }
  };

  // ---- load aspect split data ----
  const loadAspectSplit = async (sentiment) => {
    try {
      setLoadingAspectSplit(true);
      const data = await getAspectSentimentSplit(start, end, true, true); // Include others
      
      console.log('Raw API response:', data); // Debug log
      
      // Transform the API response to match our expected format
      let filteredData = [];
      
      if (data?.labels && data?.counts && data?.percent) {
        // Transform the API response structure
        const labels = data.labels;
        const counts = data.counts[sentiment.toLowerCase()] || [];
        const percentages = data.percent[sentiment.toLowerCase()] || [];
        
        filteredData = labels.map((aspect, index) => ({
          aspect: aspect === 'others' ? 'Others' : aspect.charAt(0).toUpperCase() + aspect.slice(1),
          count: counts[index] || 0,
          percentage: percentages[index] || 0,
          sentiment: sentiment.toLowerCase()
        }));
      } else {
        // Fallback for other data structures
        if (data?.aspects) {
          filteredData = data.aspects.filter(aspect => 
            aspect.sentiment?.toLowerCase() === sentiment.toLowerCase()
          );
        } else if (Array.isArray(data)) {
          filteredData = data.filter(aspect => 
            aspect.sentiment?.toLowerCase() === sentiment.toLowerCase()
          );
        } else if (data?.[sentiment.toLowerCase()]) {
          filteredData = data[sentiment.toLowerCase()] || [];
        } else {
          const allAspects = Object.values(data || {}).flat();
          filteredData = allAspects.filter(aspect => 
            aspect.sentiment?.toLowerCase() === sentiment.toLowerCase()
          );
        }
      }
      
      console.log('Transformed data for', sentiment, ':', filteredData); // Debug log
      
      setAspectSplitModal({
        isOpen: true,
        sentiment: sentiment,
        data: filteredData
      });
    } catch (error) {
      console.error("Failed to load aspect split:", error);
      setErr("Failed to load aspect breakdown");
    } finally {
      setLoadingAspectSplit(false);
    }
  };

  // Format date function for chart labels
  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    switch (timePeriod) {
      case "daily":
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      case "weekly":
        return `Week ${date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;
      case "monthly":
        return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
      default:
        return dateStr;
    }
  };

  const trendLineData = useMemo(() => {
    if (!trend || trend.length === 0) return { labels: [], datasets: [] };

    // Aggregate data based on time period
    const aggregateData = (data, period) => {
      const grouped = {};
      
      data.forEach(item => {
        const date = new Date(item.date);
        let key;
        
        switch (period) {
          case "weekly":
            // Group by week (Monday start)
            const weekStart = new Date(date);
            weekStart.setDate(date.getDate() - date.getDay() + 1);
            key = weekStart.toISOString().split('T')[0];
            break;
          case "monthly":
            // Group by month
            key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            break;
          default:
            // Daily - no aggregation needed
            key = item.date;
        }
        
        if (!grouped[key]) {
          grouped[key] = {
            date: key,
            positive: 0,
            neutral: 0,
            negative: 0,
            count: 0
          };
        }
        
        grouped[key].positive += item.positive || 0;
        grouped[key].neutral += item.neutral || 0;
        grouped[key].negative += item.negative || 0;
        grouped[key].count += 1;
      });
      
      // Calculate averages
      return Object.values(grouped).map(item => ({
        date: item.date,
        positive: item.count > 0 ? item.positive / item.count : 0,
        neutral: item.count > 0 ? item.neutral / item.count : 0,
        negative: item.count > 0 ? item.negative / item.count : 0
      })).sort((a, b) => new Date(a.date) - new Date(b.date));
    };

    // Get aggregated data
    const aggregatedTrend = timePeriod === "daily" ? trend : aggregateData(trend, timePeriod);

    // Format labels based on time period
    const labels = aggregatedTrend.map((r) => formatDate(r.date));
    const pos = aggregatedTrend.map((r) => r.positive ?? 0);
    const neu = aggregatedTrend.map((r) => r.neutral ?? 0);
    const neg = aggregatedTrend.map((r) => r.negative ?? 0);
    
    return {
      labels,
      datasets: [
        { label: "% Positive", data: pos, borderColor: "#22c55e", fill: false },
        { label: "% Neutral", data: neu, borderColor: "#facc15", fill: false },
        { label: "% Negative", data: neg, borderColor: "#ef4444", fill: false },
      ],
    };
  }, [trend, timePeriod]);

  const total = summary?.total || 0;
  const pct = summary?.percent || { positive: 0, negative: 0, neutral: 0 };

  return (
    <div className="h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-hidden">
      <div className="h-full flex flex-col px-1 py-0.5 max-w-8xl mx-auto">
      {/* Error & Loading States */}
      {err && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-4 backdrop-blur-sm">
            <div className="flex items-center space-x-3">
              <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
              <span className="text-red-300 font-semibold">{err}</span>
          </div>
        </div>
      )}

      {loading && (
          <div className="flex items-center justify-center h-full">
            <div className="flex flex-col items-center space-y-4">
              <div className="w-8 h-8 border-3 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
              <span className="text-slate-300 font-semibold text-lg">Loading analytics...</span>
          </div>
        </div>
      )}

        {!loading && (
        <>
            {/* Ultra-Compact Analytics Dashboard Header */}
            <div className="mb-0.5 pt-2 pb-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className="w-6 h-6 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-lg flex items-center justify-center shadow-lg">
                    <svg className="w-3 h-3 text-slate-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <div className="px-2 py-1">
                    <h1 className="text-sm font-bold text-white">Analytics Dashboard</h1>
                    <p className="text-slate-400 text-xs">Real-time sentiment analysis</p>
                  </div>
                </div>
                <div className="flex items-center">
                  {/* Date Range Controls */}
                  <div className="flex items-center space-x-1 bg-slate-700/30 backdrop-blur-sm rounded-md border border-slate-600/50 px-3 py-2">
                    <div className="w-1 h-1 bg-gradient-to-r from-emerald-400 to-cyan-400 rounded-full"></div>
                    <span className="text-xs text-slate-200 font-medium">Date:</span>
                    <input
                      type="date"
                      value={start || ""}
                      min={meta?.min || ""}
                      max={end || meta?.max || ""}
                      onChange={(e) => setStart(iso(e.target.value))}
                      className="px-1 py-0.5 bg-slate-800/50 border border-slate-600 rounded text-xs text-white focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all"
                    />
                    <span className="text-slate-400 text-xs">to</span>
                    <input
                      type="date"
                      value={end || ""}
                      min={start || meta?.min || ""}
                      max={meta?.max || ""}
                      onChange={(e) => setEnd(iso(e.target.value))}
                      className="px-1 py-0.5 bg-slate-800/50 border border-slate-600 rounded text-xs text-white focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Ultra-Compact Main Content Layout */}
            <div className="flex-1 grid grid-cols-1 lg:grid-cols-8 gap-0.5 min-h-0">
              {/* Left Side - Compact KPI Cards */}
              <div className="lg:col-span-1 space-y-0.5">
                {/* Total Tweets Card */}
                <div className="bg-gradient-to-br from-slate-800/60 to-slate-700/60 backdrop-blur-sm rounded-md p-3 border border-slate-600/30 shadow-md hover:shadow-emerald-500/20 hover:border-emerald-400/40 transition-all duration-500 group cursor-pointer">
                  <div className="flex items-center justify-between mb-1">
                    <div className="w-6 h-6 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-md flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                      <svg className="w-3 h-3 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <div className="text-right">
                      <p className="text-xs font-medium text-slate-400 mb-0">Total Tweets</p>
                      <p className="text-sm font-bold text-white group-hover:text-emerald-400 transition-colors duration-300">{total.toLocaleString()}</p>
                    </div>
                  </div>
              <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-500">All time</span>
                    <div className="flex items-center space-x-1">
                      <div className="w-1 h-1 bg-emerald-400 rounded-full animate-pulse"></div>
                      <span className="text-xs text-emerald-400 font-medium">Live</span>
                    </div>
                  </div>
                </div>

                {/* Ultra-Compact Sentiment Cards */}
                <div className="space-y-0.5">
                  {/* Positive Sentiment Card */}
                  <div className="bg-gradient-to-br from-slate-800/60 to-slate-700/60 backdrop-blur-sm rounded-md p-3 border border-slate-600/30 shadow-md hover:shadow-green-500/20 hover:border-green-400/40 transition-all duration-500 group cursor-pointer">
                    <div className="flex items-center justify-between mb-1">
                      <div className="w-5 h-5 bg-gradient-to-br from-green-500/20 to-emerald-500/20 rounded flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                        <svg className="w-2.5 h-2.5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                      <div className="text-right">
                        <p className="text-xs font-medium text-slate-400 mb-0">Positive</p>
                        <p className="text-sm font-bold text-white group-hover:text-green-400 transition-colors duration-300">{pct.positive ?? 0}%</p>
              </div>
            </div>
                    <div className="w-full bg-slate-700/50 rounded-full h-0.5 mb-1">
                      <div
                        className="bg-gradient-to-r from-green-500 to-emerald-500 h-0.5 rounded-full transition-all duration-500"
                        style={{ width: `${pct.positive ?? 0}%` }}
                      ></div>
                    </div>
              <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-500">Sentiment</span>
                      <div className="flex items-center space-x-1">
                        <div className="w-0.5 h-0.5 bg-green-400 rounded-full animate-pulse"></div>
                        <span className="text-xs text-green-400 font-medium">Good</span>
                      </div>
                    </div>
                    <button
                      onClick={() => loadAspectSplit('positive')}
                      disabled={loadingAspectSplit}
                      className="w-full mt-2 px-2 py-1 bg-green-500/20 hover:bg-green-500/30 text-green-400 text-xs font-medium rounded border border-green-500/30 hover:border-green-500/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loadingAspectSplit ? 'Loading...' : 'View Aspects'}
                    </button>
                </div>

                  {/* Neutral Sentiment Card */}
                  <div className="bg-gradient-to-br from-slate-800/60 to-slate-700/60 backdrop-blur-sm rounded-md p-3 border border-slate-600/30 shadow-md hover:shadow-yellow-500/20 hover:border-yellow-400/40 transition-all duration-500 group cursor-pointer">
                    <div className="flex items-center justify-between mb-1">
                      <div className="w-5 h-5 bg-gradient-to-br from-yellow-500/20 to-amber-500/20 rounded flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                        <svg className="w-2.5 h-2.5 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                      <div className="text-right">
                        <p className="text-xs font-medium text-slate-400 mb-0">Neutral</p>
                        <p className="text-sm font-bold text-white group-hover:text-yellow-400 transition-colors duration-300">{pct.neutral ?? 0}%</p>
              </div>
            </div>
                    <div className="w-full bg-slate-700/50 rounded-full h-0.5 mb-1">
                      <div
                        className="bg-gradient-to-r from-yellow-500 to-amber-500 h-0.5 rounded-full transition-all duration-500"
                        style={{ width: `${pct.neutral ?? 0}%` }}
                      ></div>
                    </div>
              <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-500">Sentiment</span>
                      <div className="flex items-center space-x-1">
                        <div className="w-0.5 h-0.5 bg-yellow-400 rounded-full animate-pulse"></div>
                        <span className="text-xs text-yellow-400 font-medium">Neutral</span>
                      </div>
                    </div>
                    <button
                      onClick={() => loadAspectSplit('neutral')}
                      disabled={loadingAspectSplit}
                      className="w-full mt-2 px-2 py-1 bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-400 text-xs font-medium rounded border border-yellow-500/30 hover:border-yellow-500/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loadingAspectSplit ? 'Loading...' : 'View Aspects'}
                    </button>
                </div>

                  {/* Negative Sentiment Card */}
                  <div className="bg-gradient-to-br from-slate-800/60 to-slate-700/60 backdrop-blur-sm rounded-md p-3 border border-slate-600/30 shadow-md hover:shadow-red-500/20 hover:border-red-400/40 transition-all duration-500 group cursor-pointer">
                    <div className="flex items-center justify-between mb-1">
                      <div className="w-5 h-5 bg-gradient-to-br from-red-500/20 to-pink-500/20 rounded flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                        <svg className="w-2.5 h-2.5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h.01M15 12h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                      </div>
                      <div className="text-right">
                        <p className="text-xs font-medium text-slate-400 mb-0">Negative</p>
                        <p className="text-sm font-bold text-white group-hover:text-red-400 transition-colors duration-300">{pct.negative ?? 0}%</p>
                      </div>
                    </div>
                    <div className="w-full bg-slate-700/50 rounded-full h-0.5 mb-1">
                      <div
                        className="bg-gradient-to-r from-red-500 to-pink-500 h-0.5 rounded-full transition-all duration-500"
                        style={{ width: `${pct.negative ?? 0}%` }}
                      ></div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-500">Sentiment</span>
                      <div className="flex items-center space-x-1">
                        <div className="w-0.5 h-0.5 bg-red-400 rounded-full animate-pulse"></div>
                        <span className="text-xs text-red-400 font-medium">Alert</span>
                      </div>
                    </div>
                    <button
                      onClick={() => loadAspectSplit('negative')}
                      disabled={loadingAspectSplit}
                      className="w-full mt-2 px-2 py-1 bg-red-500/20 hover:bg-red-500/30 text-red-400 text-xs font-medium rounded border border-red-500/30 hover:border-red-500/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loadingAspectSplit ? 'Loading...' : 'View Aspects'}
                    </button>
                  </div>
                </div>
              </div>

              {/* Middle Side - Aspect Breakdown Card */}
              {aspectSplitModal.isOpen && (
                <div className="lg:col-span-1.5 flex flex-col min-h-0">
                  <div className="bg-gradient-to-br from-slate-800/60 to-slate-700/60 backdrop-blur-sm rounded-md p-4 border border-slate-600/30 shadow-md flex flex-col min-h-0" style={{ height: 'calc(100vh - 180px)' }}>
                    <div className="flex items-center justify-between mb-4 flex-shrink-0">
                      <div className="flex items-center space-x-2">
                        <div className="w-6 h-6 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-md flex items-center justify-center">
                          <svg className="w-3 h-3 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                          </svg>
                        </div>
                        <div>
                          <h2 className="text-sm font-bold text-white">
                            {aspectSplitModal.sentiment?.charAt(0).toUpperCase() + aspectSplitModal.sentiment?.slice(1)} Aspects
                          </h2>
                          <p className="text-slate-400 text-xs">Aspect breakdown</p>
                        </div>
                      </div>
                      <button
                        onClick={() => setAspectSplitModal({ isOpen: false, sentiment: null, data: null })}
                        className="text-slate-400 hover:text-white transition-colors"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                    
                    <div className="flex-1 overflow-y-auto">
                      {aspectSplitModal.data && aspectSplitModal.data.length > 0 ? (
                        <div className="space-y-3">
                          {aspectSplitModal.data.map((aspect, index) => {
                            // Calculate percentage of total sentiment tweets
                            const totalSentimentTweets = aspectSplitModal.data.reduce((sum, a) => sum + (a.count || a.tweet_count || a.total || 0), 0);
                            const aspectCount = aspect.count || aspect.tweet_count || aspect.total || 0;
                            const percentageOfTotal = totalSentimentTweets > 0 ? (aspectCount / totalSentimentTweets) * 100 : 0;
                            
                            return (
                              <div key={index} className="space-y-1">
                                <div className="flex items-center justify-between">
                                  <h4 className="font-medium text-white text-sm">{aspect.aspect || aspect.name || aspect.label || `Aspect ${index + 1}`}</h4>
                                  <div className="flex items-center space-x-2">
                                    <span className="text-xs text-slate-400">{aspectCount} tweets</span>
                                    <span className="text-sm font-bold text-white">{percentageOfTotal.toFixed(1)}%</span>
            </div>
          </div>

                                {/* Progress Bar */}
                                <div className="w-full bg-slate-600/50 rounded-full h-3 overflow-hidden">
                                  <div
                                    className={`h-3 rounded-full transition-all duration-500 ${
                                      aspectSplitModal.sentiment === 'positive' 
                                        ? 'bg-gradient-to-r from-green-500 to-emerald-500' 
                                        : aspectSplitModal.sentiment === 'negative'
                                        ? 'bg-gradient-to-r from-red-500 to-pink-500'
                                        : 'bg-gradient-to-r from-yellow-500 to-amber-500'
                                    }`}
                                    style={{ 
                                      width: `${Math.min(percentageOfTotal, 100)}%`,
                                      maxWidth: '100%'
                                    }}
                                  ></div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <div className="text-center py-8">
                          <div className="w-12 h-12 bg-slate-700/50 rounded-full flex items-center justify-center mx-auto mb-3">
                            <svg className="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                          </div>
                          <h4 className="text-sm font-semibold text-white mb-2">No Aspect Data Found</h4>
                          <p className="text-slate-400 text-xs">No aspect breakdown available for {aspectSplitModal.sentiment} sentiment.</p>
                        </div>
                      )}
                    </div>
                    
                    {/* Summary */}
                    {aspectSplitModal.data && aspectSplitModal.data.length > 0 && (
                      <div className="mt-4 pt-3 border-t border-slate-600/50 flex-shrink-0">
                        <div className="flex items-center justify-between text-xs text-slate-400">
                          <span>Total {aspectSplitModal.sentiment} tweets analyzed</span>
                          <span>{aspectSplitModal.data.reduce((sum, a) => sum + (a.count || a.tweet_count || a.total || 0), 0)} tweets</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Chart and Date Aspect Sidebar Container */}
              <div className={`flex flex-row min-h-0 gap-2 ${aspectSplitModal.isOpen ? 'lg:col-span-5.5' : selectedDateModal.isOpen ? 'lg:col-span-5.5' : 'lg:col-span-7'}`}>
                {/* Chart Section */}
                <div className={`flex flex-col min-h-0 ${selectedDateModal.isOpen ? 'flex-1' : 'w-full'}`}>
                  <div className="bg-gradient-to-br from-slate-800/60 to-slate-700/60 backdrop-blur-sm rounded-md p-1 border border-slate-600/30 shadow-md flex flex-col min-h-0" style={{ height: 'calc(100vh - 180px)' }}>
                    <div className="flex items-center justify-between mb-1 flex-shrink-0">
                      <div className="flex items-center space-x-2">
                        <div className="w-6 h-6 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-md flex items-center justify-center">
                          <svg className="w-3 h-3 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
                        <div>
                          <h2 className="text-sm font-bold text-white">Sentiment Trend</h2>
                          <p className="text-slate-400 text-xs">Track sentiment changes</p>
                        </div>
                      </div>
                    <div className="flex items-center space-x-2">
                      {/* Test Chart Click Button */}
                      <button 
                        onClick={() => {
                          console.log('Test chart click button clicked');
                          loadDateAspects('Aug 10');
                        }}
                        className="bg-blue-500 hover:bg-blue-600 text-white text-xs px-2 py-1 rounded"
                      >
                        Test Chart Click
                      </button>
                      {/* Debug Display */}
                      <div className="text-xs text-yellow-400 bg-black/50 px-2 py-1 rounded">
                        Sidebar: {selectedDateModal.isOpen ? 'OPEN' : 'CLOSED'}
                      </div>
                      {/* Time Period Selector */}
                      <div className="flex items-center space-x-1 bg-slate-700/30 backdrop-blur-sm rounded-md border border-slate-600/50 px-2 py-1">
                          <span className="text-xs text-slate-200 font-medium">View:</span>
                          <select
                            value={timePeriod}
                            onChange={(e) => setTimePeriod(e.target.value)}
                            className="px-1 py-0.5 bg-slate-800/50 border border-slate-600 rounded text-xs text-white focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all"
                          >
                            <option value="daily">Daily</option>
                            <option value="weekly">Weekly</option>
                            <option value="monthly">Monthly</option>
                          </select>
                        </div>
                        <div className="flex items-center space-x-1 bg-slate-700/50 px-1.5 py-0.5 rounded">
                          <div className="w-1 h-1 bg-emerald-400 rounded-full animate-pulse"></div>
                          <span className="text-xs text-slate-300 font-medium">Live</span>
                        </div>
                        <div className="text-xs text-slate-400 bg-slate-700/30 px-1.5 py-0.5 rounded">
                          {trend.length} pts
                        </div>
                        {loadingMore && (
                          <div className="flex items-center space-x-1 text-xs text-emerald-400">
                            <div className="w-2 h-2 border border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
                            <span>Loading...</span>
                          </div>
                        )}
                      </div>
            </div>
                    
                    <div 
                      className="flex-1 bg-slate-900/30 rounded p-0.5 border border-slate-600/20 min-h-0 overflow-x-auto overflow-y-hidden"
                      onScroll={(e) => {
                        const { scrollLeft, scrollWidth, clientWidth } = e.target;
                        // Load more data when scrolled to 80% of the content
                        if (scrollLeft + clientWidth >= scrollWidth * 0.8 && hasMoreData && !loadingMore) {
                          loadMoreData();
                        }
                      }}
                    >
              <Line 
                data={trendLineData} 
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                          layout: {
                            padding: {
                              right: hasMoreData ? 20 : 0
                            }
                          },
                          onClick: (event, elements) => {
                            if (elements.length > 0) {
                              const element = elements[0];
                              const datasetIndex = element.datasetIndex;
                              const dataIndex = element.index;
                              const selectedDate = trendLineData.labels[dataIndex];
                              
                              loadDateAspects(selectedDate);
                            }
                          },
                          interaction: {
                            intersect: false,
                            mode: 'index'
                          },
                  plugins: {
                    legend: {
                      position: 'top',
                      labels: {
                        color: '#e2e8f0',
                        font: {
                                  size: 9,
                          weight: 'bold'
                                },
                                padding: 8,
                                usePointStyle: true,
                                pointStyle: 'circle'
                              }
                            },
                            tooltip: {
                              backgroundColor: 'rgba(15, 23, 42, 0.95)',
                              titleColor: '#f1f5f9',
                              bodyColor: '#cbd5e1',
                              borderColor: 'rgba(148, 163, 184, 0.2)',
                              borderWidth: 1,
                              cornerRadius: 4,
                              displayColors: true,
                              padding: 4
                    }
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                              max: 80,
                      grid: {
                                color: 'rgba(148, 163, 184, 0.1)',
                                drawBorder: false
                      },
                      ticks: {
                        color: '#94a3b8',
                        font: {
                                  size: 8
                                },
                                padding: 2
                      }
                    },
                    x: {
                      grid: {
                        display: false
                      },
                      ticks: {
                        color: '#94a3b8',
                        font: {
                                  size: 8
                                },
                                padding: 2
                      }
                    }
                  }
                }}
              />
                    </div>
                  </div>
                </div>

                {/* Date Aspect Breakdown Sidebar - Using Same Pattern as Aspect Split */}
                {selectedDateModal.isOpen && (
                  <div className="lg:col-span-1.5 flex flex-col min-h-0">
                    <div className="bg-gradient-to-br from-slate-800/60 to-slate-700/60 backdrop-blur-sm rounded-md p-4 border border-slate-600/30 shadow-md flex flex-col min-h-0" style={{ height: 'calc(100vh - 180px)' }}>
                      <div className="flex items-center justify-between mb-4 flex-shrink-0">
                        <div className="flex items-center space-x-2">
                          <div className="w-6 h-6 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-md flex items-center justify-center">
                            <svg className="w-3 h-3 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                          </div>
                          <div>
                            <h3 className="text-sm font-bold text-white">Date Aspect Breakdown</h3>
                            <p className="text-slate-400 text-xs">Date: {selectedDateModal.formattedDate || selectedDateModal.date}</p>
                          </div>
                        </div>
                        <button
                          onClick={() => setSelectedDateModal({ isOpen: false, date: null, formattedDate: null, data: null })}
                          className="w-6 h-6 bg-slate-700/50 hover:bg-slate-600/50 rounded-md flex items-center justify-center transition-colors"
                        >
                          <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>

                      <div className="flex-1 bg-slate-900/30 rounded p-2 border border-slate-600/20 min-h-0 overflow-y-auto">
                        <div className="text-center py-8">
                          <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                            <svg className="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                          </div>
                          <h4 className="text-red-400 font-bold text-lg mb-2">SIDEBAR IS WORKING!</h4>
                          <p className="text-slate-300 text-sm mb-4">This sidebar is now functional</p>
                          <p className="text-slate-400 text-xs">Selected Date: {selectedDateModal.formattedDate || selectedDateModal.date}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
            </div>
          </div>
        </>
      )}
      </div>
    </div>
  );
}