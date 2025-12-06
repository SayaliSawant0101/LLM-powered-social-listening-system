// frontend/src/api.js
import axios from "axios";

// ----------------------------------------
// Base URL configuration
// ----------------------------------------

// Your Railway production URL
const RAILWAY_URL =
  "https://llm-powered-social-listening-system-production.up.railway.app";

// Detect if we're running on Netlify (production)
const isNetlifyProd = typeof window !== "undefined" &&
  window.location.hostname.endsWith("netlify.app");

// Choose BASE_URL:
// 1) Prefer env variable (Vite / Netlify)
// 2) If on Netlify and env is missing, fall back to Railway URL
// 3) Otherwise (local dev), fall back to localhost
const RAW_BASE_URL =
  import.meta?.env?.VITE_API_BASE_URL ||
  (isNetlifyProd ? RAILWAY_URL : "http://localhost:3001");

// Remove any trailing slash just to be safe
const BASE_URL = RAW_BASE_URL.replace(/\/+$/, "");

// Debug: confirm what the app is actually using
console.log("🛰 API BASE_URL =", BASE_URL);

// Main API instance
const API = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
});

// Special API instance for long-running operations like theme generation
const LONG_API = axios.create({
  baseURL: BASE_URL,
  timeout: 1200000, // 20 minutes for theme generation
});

// --- Sentiment ---
export async function getMeta() {
  const { data } = await API.get("/");
  return data?.date_range || null;
}

export async function getSummary(start, end) {
  const { data } = await API.get("/api/sentiment/summary", {
    params: { start, end },
  });
  return data;
}

export async function getTrend(
  start,
  end,
  period = "daily",
  offset = 0,
  limit = 0
) {
  const { data } = await API.get("/api/sentiment/trend", {
    params: { start, end, period, offset, limit },
  });
  return data;
}

// --- Aspects ---
export async function getAspectSummary(start, end, asPercent = false) {
  const { data } = await API.get("/api/aspects/summary", {
    params: { start, end, as_percent: asPercent },
  });
  return data;
}

export async function getAspectAvgScores(start, end) {
  const { data } = await API.get("/api/aspects/avg-scores", {
    params: { start, end },
  });
  return data;
}

// --- Aspect × Sentiment (Stacked Bar) ---
export async function getAspectSentimentSplit(
  start,
  end,
  asPercent = false,
  includeOthers = false
) {
  const { data } = await API.get("/api/aspects/sentiment-split", {
    params: {
      start,
      end,
      as_percent: asPercent,
      include_others: includeOthers,
    },
  });
  return data;
}

// Get raw aspect data for calculating "Others" category
export async function getRawAspectData(start, end) {
  const { data } = await API.get("/api/aspects/sentiment-split", {
    params: { start, end, as_percent: false },
  });
  return data;
}

// Get sample tweets for specific aspect and sentiment
export async function getSampleTweets(
  start,
  end,
  aspect,
  sentiment,
  limit = 10
) {
  const { data } = await API.get("/api/tweets/sample", {
    params: { start, end, aspect, sentiment, limit },
  });
  return data.tweets || [];
}

// --- Themes (dynamic clustering + summaries) ---
export async function fetchThemes({
  start = null,
  end = null,
  n_clusters = null, // Auto-detect if null
  emb_model = "sentence-transformers/all-MiniLM-L6-v2",
  parquet = null,
  max_rows = null,
} = {}) {
  const params = {};
  if (start) params.start = start;
  if (end) params.end = end;
  if (n_clusters !== null) params.n_clusters = n_clusters;
  if (emb_model) params.emb_model = emb_model;
  if (parquet) params.parquet = parquet;
  if (typeof max_rows === "number" && Number.isFinite(max_rows)) {
    params.max_rows = max_rows;
  }

  const { data } = await LONG_API.get("/api/themes", { params });
  return data; // { updated_at, themes: [{id, name, summary, tweet_count}] }
}

// --- Raw Data Downloads ---
export async function downloadRawTweets(start, end, format = "csv") {
  const response = await API.get("/api/tweets/raw", {
    params: { start, end, format },
    responseType: "blob",
  });

  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `raw_tweets_${start}_to_${end}.${format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

export async function downloadSentimentReport(start, end, format = "pdf") {
  const response = await API.get("/api/reports/sentiment", {
    params: { start, end, format },
    responseType: "blob",
  });

  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `sentiment_report_${start}_to_${end}.${format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

export async function downloadAspectReport(start, end, format = "pdf") {
  const response = await API.get("/api/reports/aspects", {
    params: { start, end, format },
    responseType: "blob",
  });

  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `aspect_report_${start}_to_${end}.${format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

export async function downloadThemeReport(start, end, format = "pdf") {
  const response = await API.get("/api/reports/themes", {
    params: { start, end, format },
    responseType: "blob",
  });

  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `theme_report_${start}_to_${end}.${format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

export async function downloadDashboardReport(start, end, format = "pdf") {
  const response = await API.get("/api/reports/dashboard", {
    params: { start, end, format },
    responseType: "blob",
  });

  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `analytics_dashboard_${start || "all"}_to_${
    end || "all"
  }.${format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

export async function downloadThemeTweetsReport(themeId, start, end) {
  const response = await API.get(`/api/reports/theme/${themeId}`, {
    params: { start, end, limit: 200 },
    responseType: "blob",
  });

  const blob = new Blob([response.data], { type: "text/html" });
  const url = window.URL.createObjectURL(blob);

  // Open in new tab instead of downloading
  window.open(url, "_blank");

  // Clean up the URL after a delay to free memory
  setTimeout(() => {
    window.URL.revokeObjectURL(url);
  }, 10000);
}
