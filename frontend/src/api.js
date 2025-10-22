// frontend/src/api.js
import axios from "axios";

// Prefer env if provided; otherwise fall back to localhost
const BASE_URL = import.meta?.env?.VITE_API_BASE || "http://127.0.0.1:8000";

// If you set up a Vite proxy, change baseURL to "/api"
const API = axios.create({
  baseURL: BASE_URL, // or "http://localhost:8000"
  timeout: 15000,
});

// Special API instance for long-running operations like theme generation
const LONG_API = axios.create({
  baseURL: BASE_URL,
  timeout: 180000, // 3 minutes for theme generation (optimized)
});

// --- Sentiment ---
export async function getMeta() {
  const { data } = await API.get("/");
  // prefer aspect range if needed, but keep existing behavior
  return data?.date_range || null;
}

export async function getSummary(start, end) {
  const { data } = await API.get("/sentiment/summary", { params: { start, end } });
  return data;
}

export async function getTrend(start, end, period = "daily", offset = 0, limit = 50) {
  const { data } = await API.get("/sentiment/trend", { 
    params: { start, end, period, offset, limit } 
  });
  return data;
}

// --- Aspects ---
export async function getAspectSummary(start, end, asPercent = false) {
  const { data } = await API.get("/aspects/summary", {
    params: { start, end, as_percent: asPercent },
  });
  return data;
}

export async function getAspectAvgScores(start, end) {
  const { data } = await API.get("/aspects/avg-scores", { params: { start, end } });
  return data;
}

// --- Aspect × Sentiment (Stacked Bar) ---
export async function getAspectSentimentSplit(start, end, asPercent = false, includeOthers = false) {
  const { data } = await API.get("/aspects/sentiment-split", {
    params: { start, end, as_percent: asPercent, include_others: includeOthers },
  });
  return data;
}

// Get raw aspect data for calculating "Others" category
export async function getRawAspectData(start, end) {
  const { data } = await API.get("/aspects/sentiment-split", {
    params: { start, end, as_percent: false }
  });
  return data;
}

// --- Themes (dynamic clustering + summaries) ---
export async function fetchThemes({
  start = null,
  end = null,
  n_clusters = 6, // Limited to 6 themes max
  emb_model = "sentence-transformers/all-MiniLM-L6-v2",
} = {}) {
  // Build params without null/empty values
  const params = {};
  if (start) params.start = start;
  if (end) params.end = end;
  if (n_clusters) params.n_clusters = n_clusters;
  if (emb_model) params.emb_model = emb_model;

  const { data } = await LONG_API.get("/themes", { params });
  return data; // { updated_at, themes: [{id, name, summary, tweet_count}] }
}
