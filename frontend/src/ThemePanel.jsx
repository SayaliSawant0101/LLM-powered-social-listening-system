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

  // View theme tweets as PDF function
  const viewThemeTweetsAsPDF = async (theme) => {
    try {
      // Fetch tweets for this specific theme
      const response = await fetch(`http://localhost:8000/themes/${theme.id}/tweets?limit=200`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch tweets: ${response.status}`);
      }
      
      const tweetsData = await response.json();
      const tweets = tweetsData.items || [];
      
      if (tweets.length === 0) {
        alert('No tweets found for this theme');
        return;
      }
      
      // Calculate sentiment split
      const sentimentCounts = tweets.reduce((acc, tweet) => {
        const sentiment = tweet.sentiment_label || 'unknown';
        acc[sentiment] = (acc[sentiment] || 0) + 1;
        return acc;
      }, {});
      
      const totalTweets = tweets.length;
      const positiveCount = sentimentCounts.positive || 0;
      const negativeCount = sentimentCounts.negative || 0;
      const neutralCount = sentimentCounts.neutral || 0;
      const unknownCount = sentimentCounts.unknown || 0;
      
      // Create PDF content
      const pdfContent = `
        <html>
          <head>
            <title>Theme Tweets Report</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 20px; }
              h1 { color: #333; text-align: center; }
              h2 { color: #666; border-bottom: 2px solid #ddd; }
              .tweet { margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; background: #f8f9fa; }
              .metadata { font-size: 12px; color: #666; margin-top: 5px; }
              .positive { border-left-color: #28a745; }
              .negative { border-left-color: #dc3545; }
              .neutral { border-left-color: #ffc107; }
              .unknown { border-left-color: #6c757d; }
              .theme-info { background: #e9ecef; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
              .sentiment-summary { background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
              .sentiment-bar { display: flex; margin: 5px 0; }
              .sentiment-bar-item { padding: 5px 10px; margin-right: 10px; border-radius: 4px; font-weight: bold; }
              .sentiment-positive { background: #d4edda; color: #155724; }
              .sentiment-negative { background: #f8d7da; color: #721c24; }
              .sentiment-neutral { background: #fff3cd; color: #856404; }
              .sentiment-unknown { background: #e2e3e5; color: #383d41; }
            </style>
          </head>
          <body>
            <h1>Theme Tweets Report</h1>
            
            <div class="theme-info">
              <h2>Theme Information</h2>
              <p><strong>Theme Name:</strong> ${theme.name || `Theme ${theme.id}`}</p>
              <p><strong>Theme Summary:</strong> ${theme.summary || "No summary available"}</p>
              <p><strong>Total Tweets:</strong> ${theme.tweet_count || 0}</p>
              <p><strong>Date Range:</strong> ${start || 'All time'} to ${end || 'All time'}</p>
            </div>
            
            <div class="sentiment-summary">
              <h2>Sentiment Distribution</h2>
              <p><strong>Total Tweets Analyzed:</strong> ${totalTweets}</p>
              <div class="sentiment-bar">
                <div class="sentiment-bar-item sentiment-positive">Positive: ${positiveCount} (${Math.round((positiveCount/totalTweets)*100)}%)</div>
                <div class="sentiment-bar-item sentiment-negative">Negative: ${negativeCount} (${Math.round((negativeCount/totalTweets)*100)}%)</div>
                <div class="sentiment-bar-item sentiment-neutral">Neutral: ${neutralCount} (${Math.round((neutralCount/totalTweets)*100)}%)</div>
                ${unknownCount > 0 ? `<div class="sentiment-bar-item sentiment-unknown">Unknown: ${unknownCount} (${Math.round((unknownCount/totalTweets)*100)}%)</div>` : ''}
              </div>
            </div>
            
            <h2>All Tweets (${tweets.length} total)</h2>
            
            ${tweets.map((tweet, index) => `
              <div class="tweet ${tweet.sentiment_label || 'unknown'}">
                <strong>Tweet #${index + 1}</strong><br>
                ${tweet.text || tweet.text_clean || 'No text available'}<br>
                <div class="metadata">
                  <strong>Sentiment:</strong> ${tweet.sentiment_label || 'Unknown'} | 
                  <strong>Date:</strong> ${tweet.date || tweet.createdat || 'Unknown'} |
                  <strong>Aspect:</strong> ${tweet.aspect_dominant || 'Unknown'}
                </div>
              </div>
            `).join('')}
          </body>
        </html>
      `;
      
      // Open in new tab for viewing and trigger download
      const viewWindow = window.open('', '_blank');
      viewWindow.document.write(pdfContent);
      viewWindow.document.close();
      
      // Trigger print dialog for PDF download
      setTimeout(() => {
        viewWindow.print();
      }, 500);
      
      console.log('Theme tweets PDF opened and download triggered');
    } catch (error) {
      console.error('Failed to view theme tweets:', error);
      alert('Failed to view theme tweets. Please try again.');
    }
  };

  return (
    <div className="space-y-4">
      {/* Modern Controls */}
      <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl p-4 border border-slate-600/30 shadow-xl">
        <div className="flex items-center space-x-6 mb-4">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-gradient-to-r from-emerald-400 to-cyan-400 rounded-full"></div>
            <span className="font-semibold text-slate-200 text-sm">Theme Configuration</span>
          </div>
          
          <div className="flex items-center space-x-2">
            <label className="text-xs font-medium text-slate-400"># Themes</label>
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
        
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex items-center space-x-4">
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
                  <div className="flex items-center space-x-2">
                    <span className="font-bold text-white">
                      {t.tweet_count ?? 0} tweets
                      {data?.themes?.length && (
                        <span className="text-slate-400 ml-1">
                          ({Math.round(((t.tweet_count ?? 0) / data.themes.reduce((sum, theme) => sum + (theme.tweet_count ?? 0), 0)) * 100)}%)
                        </span>
                      )}
                    </span>
                    <button
                      onClick={() => viewThemeTweetsAsPDF(t)}
                      className="px-2 py-1 bg-slate-600 hover:bg-slate-700 text-white text-xs rounded transition-colors flex items-center space-x-1"
                      title="View Theme Tweets"
                    >
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                      </svg>
                      <span>View Tweets</span>
                    </button>
                  </div>
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
