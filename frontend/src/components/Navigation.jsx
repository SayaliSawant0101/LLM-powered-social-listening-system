// frontend/src/components/Navigation.jsx
import React from "react";
import { Link, useLocation } from "react-router-dom";
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

export default function Navigation() {
  const location = useLocation();
  const { start, end, meta, setStart, setEnd } = useDate();

  const navItems = [
    {
      path: "/",
      label: "Dashboard",
      description: "Overview & Sentiment Trends"
    },
    {
      path: "/aspect-analysis",
      label: "Aspect Analysis",
      description: "Aspect × Sentiment Breakdown"
    },
    {
      path: "/theme-analysis",
      label: "Theme Analysis",
      description: "AI-Generated Theme Clusters"
    },
    {
      path: "/ai-insights",
      label: "AI Insights",
      description: "Executive Summary & Briefs"
    }
  ];

  return (
    <>
      {/* Top Navigation - Branding Only */}
      <nav className="bg-slate-800/95 backdrop-blur-md border-b border-slate-700 sticky top-0 z-50">
        <div className="max-w-8xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-2xl flex items-center justify-center shadow-lg">
                <span className="text-slate-900 font-bold text-xl">W</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Walmart Social Intelligence</h1>
                <p className="text-slate-400">Advanced Analytics Platform</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 bg-slate-700/50 px-3 py-2 rounded-xl">
                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                <span className="text-sm text-slate-300 font-medium">Live Data</span>
              </div>
              <div className="text-sm text-slate-400 bg-slate-700/30 px-3 py-2 rounded-xl">
                API Connected
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Sidebar Navigation */}
      <div className="fixed left-0 top-20 h-screen w-64 bg-slate-800/95 backdrop-blur-md border-r border-slate-700 z-40">
        <div className="p-4">
          <div className="space-y-2">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-300 ${
                  location.pathname === item.path
                    ? "bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 text-emerald-400 border border-emerald-400/30 shadow-lg shadow-emerald-400/10"
                    : "text-slate-400 hover:text-white hover:bg-slate-700/30"
                }`}
              >
                <div className="w-8 h-8 bg-slate-700/50 rounded-lg flex items-center justify-center">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <div>
                  <div className="font-semibold text-sm">{item.label}</div>
                  <div className="text-xs text-slate-500">{item.description}</div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
