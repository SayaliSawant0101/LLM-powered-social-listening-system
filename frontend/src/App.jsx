// frontend/src/App.jsx
import { useEffect, useMemo, useState } from "react";
import { Bar, Line } from "react-chartjs-2";
import "chart.js/auto";
import {
  getMeta,
  getSummary,
  getTrend,
  getAspectSummary,
  getAspectSentimentSplit,   // ⬅ NEW
} from "./api";

function iso(x) {
  if (!x) return "";
  const d = new Date(x);
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

export default function App() {
  const [meta, setMeta] = useState(null);  // {min, max}
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");

  // sentiment
  const [summary, setSummary] = useState(null);
  const [trend, setTrend] = useState([]);

  // aspects (simple distribution)
  const [aspectSummary, setAspectSummary] = useState(null);
  const [asPercent, setAsPercent] = useState(false);

  // aspects × sentiment (stacked)
  const [split, setSplit] = useState(null);
  const [splitAsPercent, setSplitAsPercent] = useState(false);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  // load meta once
  useEffect(() => {
    (async () => {
      try {
        const mr = await getMeta();
        setMeta(mr);
        setStart(mr?.min || "");
        setEnd(mr?.max || "");
      } catch {
        setErr("Failed to load metadata");
      }
    })();
  }, []);

  // load sentiment summary + trend
  useEffect(() => {
    if (!start || !end) return;
    (async () => {
      try {
        setLoading(true);
        setErr("");
        const [s, t] = await Promise.all([getSummary(start, end), getTrend(start, end)]);
        setSummary(s);
        setTrend(t?.trend || []);
      } catch {
        setErr("Failed to load data. Is the API running on :8000?");
      } finally {
        setLoading(false);
      }
    })();
  }, [start, end]);

  // load aspects distribution
  useEffect(() => {
    if (!start || !end) return;
    (async () => {
      try {
        const res = await getAspectSummary(start, end, asPercent);
        setAspectSummary(res);
      } catch (e) {
        console.error("Failed to load aspects:", e);
      }
    })();
  }, [start, end, asPercent]);

  // load aspects × sentiment split (stacked)
  useEffect(() => {
    if (!start || !end) return;
    (async () => {
      try {
        const s = await getAspectSentimentSplit(start, end, splitAsPercent);
        setSplit(s);
      } catch (e) {
        console.error("Failed to load aspect split", e);
      }
    })();
  }, [start, end, splitAsPercent]);

  // ----- chart data -----
  const sentimentBarData = useMemo(() => {
    const counts = summary?.counts || {};
    const labels = ["positive", "neutral", "negative"];
    const values = labels.map((k) => counts[k] || 0);
    return {
      labels: labels.map((s) => s[0].toUpperCase() + s.slice(1)),
      datasets: [
        {
          label: "Tweet Count",
          data: values,
          backgroundColor: ["#22c55e", "#facc15", "#ef4444"],
        },
      ],
    };
  }, [summary]);

  const trendLineData = useMemo(() => {
    const labels = trend.map((r) => r.date);
    const pos = trend.map((r) => r.positive ?? 0);
    const neu = trend.map((r) => r.neutral ?? 0);
    const neg = trend.map((r) => r.negative ?? 0);
    return {
      labels,
      datasets: [
        { label: "% Positive", data: pos, borderColor: "#22c55e", fill: false },
        { label: "% Neutral", data: neu, borderColor: "#facc15", fill: false },
        { label: "% Negative", data: neg, borderColor: "#ef4444", fill: false },
      ],
    };
  }, [trend]);

  const aspectBarData = useMemo(() => {
    if (!aspectSummary) return { labels: [], datasets: [] };
    const labels = aspectSummary.labels || ["pricing", "delivery", "returns", "staff", "app/ux"];
    const seriesObj = aspectSummary.series || aspectSummary.counts || {};
    const values = labels.map((l) => seriesObj[l] ?? 0);
    return {
      labels: labels.map((t) => t.toUpperCase()),
      datasets: [
        {
          label: asPercent ? "Aspect Share (%)" : "Aspect Count",
          data: values,
          backgroundColor: ["#3b82f6", "#22c55e", "#f97316", "#a855f7", "#ef4444"],
        },
      ],
    };
  }, [aspectSummary, asPercent]);

  // stacked bar: aspect × sentiment
  const stackedBarData = useMemo(() => {
    if (!split) return { labels: [], datasets: [] };
    const labels = split.labels || [];
    const series = splitAsPercent ? split.percent : split.counts; // {positive:[], neutral:[], negative:[]}
    return {
      labels: labels.map((t) => t.toUpperCase()),
      datasets: [
        { label: splitAsPercent ? "% Positive" : "Positive", data: series.positive || [], backgroundColor: "#22c55e", stack: "s" },
        { label: splitAsPercent ? "% Neutral"  : "Neutral",  data: series.neutral  || [], backgroundColor: "#facc15", stack: "s" },
        { label: splitAsPercent ? "% Negative" : "Negative", data: series.negative || [], backgroundColor: "#ef4444", stack: "s" },
      ],
    };
  }, [split, splitAsPercent]);

  const stackedBarOptions = {
    responsive: true,
    plugins: { legend: { position: "top" } },
    scales: {
      x: { stacked: true },
      y: {
        stacked: true,
        suggestedMax: splitAsPercent ? 100 : undefined,
        ticks: { callback: (v) => (splitAsPercent ? `${v}%` : v) },
      },
    },
  };

  const total = summary?.total || 0;
  const pct = summary?.percent || {};

  return (
    <div style={{ minHeight: "100vh", background: "#f9fafb", padding: "24px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 12 }}>🧠 Walmart Sentiment Dashboard</h1>

      {/* Controls */}
      <div
        style={{
          display: "flex",
          gap: 16,
          alignItems: "center",
          marginBottom: 16,
          background: "#fff",
          padding: 16,
          borderRadius: 12,
          boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
        }}
      >
        <div><strong>Date range:</strong></div>
        <input
          type="date"
          value={start || ""}
          min={meta?.min || ""}
          max={end || meta?.max || ""}
          onChange={(e) => setStart(iso(e.target.value))}
          style={{ padding: 8, borderRadius: 8, border: "1px solid #e5e7eb" }}
        />
        <span>to</span>
        <input
          type="date"
          value={end || ""}
          min={start || meta?.min || ""}
          max={meta?.max || ""}
          onChange={(e) => setEnd(iso(e.target.value))}
          style={{ padding: 8, borderRadius: 8, border: "1px solid #e5e7eb" }}
        />
        <div style={{ marginLeft: "auto", fontSize: 12, color: "#6b7280" }}>
          API: http://127.0.0.1:8000
        </div>
      </div>

      {err && <div style={{ color: "#b91c1c", marginBottom: 12 }}>{err}</div>}
      {loading && <div>Loading...</div>}

      {!loading && summary && (
        <>
          {/* KPI cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 16 }}>
            <div style={{ background: "#fff", padding: 16, borderRadius: 12, boxShadow: "0 1px 3px rgba(0,0,0,0.08)" }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>Total Tweets</div>
              <div style={{ fontSize: 24, fontWeight: 700 }}>{total}</div>
            </div>
            <div style={{ background: "#fff", padding: 16, borderRadius: 12, boxShadow: "0 1px 3px rgba(0,0,0,0.08)" }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>% Positive</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: "#16a34a" }}>{pct.positive ?? 0}%</div>
            </div>
            <div style={{ background: "#fff", padding: 16, borderRadius: 12, boxShadow: "0 1px 3px rgba(0,0,0,0.08)" }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>% Neutral</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: "#a16207" }}>{pct.neutral ?? 0}%</div>
            </div>
            <div style={{ background: "#fff", padding: 16, borderRadius: 12, boxShadow: "0 1px 3px rgba(0,0,0,0.08)" }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>% Negative</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: "#dc2626" }}>{pct.negative ?? 0}%</div>
            </div>
          </div>

          {/* Sentiment distribution */}
          <div style={{ background: "#fff", padding: 16, borderRadius: 12, boxShadow: "0 1px 3px rgba(0,0,0,0.08)", marginBottom: 16 }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>Sentiment Distribution</h2>
            <Bar data={sentimentBarData} />
          </div>

          {/* Daily sentiment trend */}
          <div style={{ background: "#fff", padding: 16, borderRadius: 12, boxShadow: "0 1px 3px rgba(0,0,0,0.08)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>Daily Sentiment (% Share)</h2>
            <Line data={trendLineData} />
          </div>

          {/* Aspect distribution */}
          <div style={{ background: "#fff", padding: 16, borderRadius: 12, boxShadow: "0 1px 3px rgba(0,0,0,0.08)", marginTop: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <h2 style={{ fontSize: 18, fontWeight: 600 }}>Aspect Distribution {asPercent ? "(%)" : "(Count)"}</h2>
              <label style={{ fontSize: 14 }}>
                <input
                  type="checkbox"
                  checked={asPercent}
                  onChange={(e) => setAsPercent(e.target.checked)}
                  style={{ marginRight: 8 }}
                />
                Show as %
              </label>
            </div>
            <Bar data={aspectBarData} />
          </div>

          {/* Aspect × Sentiment stacked */}
          <div style={{ background: "#fff", padding: 16, borderRadius: 12, boxShadow: "0 1px 3px rgba(0,0,0,0.08)", marginTop: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <h2 style={{ fontSize: 18, fontWeight: 600 }}>Aspect × Sentiment (Stacked)</h2>
              <label style={{ fontSize: 14 }}>
                <input
                  type="checkbox"
                  checked={splitAsPercent}
                  onChange={(e) => setSplitAsPercent(e.target.checked)}
                  style={{ marginRight: 8 }}
                />
                Show as %
              </label>
            </div>
            <Bar data={stackedBarData} options={stackedBarOptions} />
          </div>
        </>
      )}
    </div>
  );
}
