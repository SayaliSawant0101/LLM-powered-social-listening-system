// frontend/src/components/Navigation.jsx
import React, { useId } from "react";
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

// Right-side brand graphic removed per request

function HeaderBrand() {
  const sparkGradientId = useId();
  const blueGradientId = useId();
  const badgeShadowId = useId();

  return (
    <div className="flex items-center space-x-3">
      <svg
        viewBox="0 0 64 64"
        className="h-11 w-11 drop-shadow-sm"
        role="img"
        aria-label="Walmart spark"
      >
        <defs>
          <radialGradient id={badgeShadowId} cx="32" cy="32" r="28" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#1e293b" stopOpacity="0.08" />
          </radialGradient>
          <linearGradient id={blueGradientId} x1="8" y1="8" x2="56" y2="56" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#1d4ed8" />
            <stop offset="100%" stopColor="#1e3a8a" />
          </linearGradient>
          <linearGradient id={sparkGradientId} x1="0" y1="-8" x2="0" y2="8" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#fbbf24" />
            <stop offset="100%" stopColor="#f59e0b" />
          </linearGradient>
        </defs>

        <circle cx="32" cy="32" r="30" fill="#0f172a" />
        <circle cx="32" cy="32" r="26" fill={`url(#${badgeShadowId})`} />
        <circle cx="32" cy="32" r="22" fill={`url(#${blueGradientId})`} />

        <g transform="translate(32 32)">
          {[0, 60, 120, 180, 240, 300].map((angle, idx) => (
            <rect
              key={`spark-${idx}`}
              x="-2.2"
              y="-15"
              width="4.4"
              height="11"
              rx="2.2"
              transform={`rotate(${angle})`}
              fill={`url(#${sparkGradientId})`}
            />
          ))}
        </g>
      </svg>

      <div className="flex flex-col leading-tight">
        <span className="text-[1.05rem] font-semibold tracking-[0.02em] text-slate-50">
          <span className="text-[#0071ce]">Walmart</span>
          <span className="ml-1 text-slate-100/90">Social Intelligence</span>
        </span>
        <span className="text-[0.6rem] uppercase tracking-[0.32em] text-slate-400">
          Advanced Analytics Platform
        </span>
      </div>
    </div>
  );
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
    },
    {
      path: "/raw-data",
      label: "Raw Data",
      description: "Download Raw Data & Reports"
    }
  ];

  return (
    <>
      {/* Top Navigation - Branding Only */}
      <nav className="bg-slate-800/95 backdrop-blur-md border-b border-slate-700 sticky top-0 z-50">
        <div className="w-full px-6 py-4">
          <div className="flex items-center justify-between">
            <HeaderBrand />
            <div className="flex items-center justify-end space-x-4">
              {/* Right-side status area intentionally left blank */}
            </div>
          </div>
        </div>
      </nav>

      {/* Sidebar Navigation */}
      <div className="fixed left-0 top-20 h-screen w-64 bg-slate-800/95 backdrop-blur-md border-r border-slate-700 z-40">
        <div className="p-4">
          <div className="space-y-2">
            {navItems.map((item) => {
              // Define icons for each page
              const getIcon = (path) => {
                switch (path) {
                  case "/":
                    return (
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5a2 2 0 012-2h4a2 2 0 012 2v2H8V5z" />
                      </svg>
                    );
                  case "/aspect-analysis":
                    return (
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    );
                  case "/theme-analysis":
                    return (
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                      </svg>
                    );
                  case "/ai-insights":
                    return (
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                    );
                  case "/raw-data":
                    return (
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    );
                  default:
                    return (
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    );
                }
              };

              return (
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
                    {getIcon(item.path)}
                  </div>
                  <div>
                    <div className="font-semibold text-sm">{item.label}</div>
                    <div className="text-xs text-slate-500">{item.description}</div>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </>
  );
}
