// frontend/src/contexts/DateContext.jsx
import React, { createContext, useContext, useState, useEffect } from "react";
import { getMeta } from "../api";

const DateContext = createContext();

export function DateProvider({ children }) {
  const [meta, setMeta] = useState(null);
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [loading, setLoading] = useState(true);

  // Load meta once on app start
  useEffect(() => {
    const loadMeta = async () => {
      try {
        console.log('Loading metadata from API...');
        const mr = await getMeta();
        console.log('Metadata loaded:', mr);
        // getMeta returns { date_range: { min, max } }
        const dateRange = mr?.date_range || mr;
        setMeta(dateRange);
        setStart(dateRange?.min || "");
        setEnd(dateRange?.max || "");
        console.log('Date range set:', dateRange?.min, 'to', dateRange?.max);
      } catch (error) {
        console.error("Failed to load metadata:", error);
        console.error("Error details:", error.response?.data || error.message);
        // If API fails, don't set fallback dates - let user see the error
        console.error("Failed to load metadata - no date range available");
      } finally {
        setLoading(false);
      }
    };

    // Add timeout to prevent hanging
    const timeoutId = setTimeout(() => {
      console.log('Metadata loading timeout - waiting for API');
      // Don't set fallback dates - wait for API response
      setLoading(false);
    }, 10000); // Increased timeout to 10 seconds

    loadMeta();

    return () => clearTimeout(timeoutId);
  }, []);

  const value = {
    meta,
    start,
    end,
    setStart,
    setEnd,
    loading
  };

  return (
    <DateContext.Provider value={value}>
      {children}
    </DateContext.Provider>
  );
}

export function useDate() {
  const context = useContext(DateContext);
  if (context === undefined) {
    throw new Error('useDate must be used within a DateProvider');
  }
  return context;
}

