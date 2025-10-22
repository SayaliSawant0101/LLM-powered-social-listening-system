// frontend/src/pages/ThemeAnalysis.jsx
import React from "react";
import ThemePanel from "../ThemePanel";

export default function ThemeAnalysis() {
  return (
    <div className="space-y-8">
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-white/20 shadow-lg">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-8 h-8 bg-rose-100 rounded-lg flex items-center justify-center">
            <span className="text-rose-600">🎨</span>
          </div>
          <h2 className="text-xl font-bold text-slate-900">Theme Analysis</h2>
        </div>
        <ThemePanel />
      </div>
    </div>
  );
}

