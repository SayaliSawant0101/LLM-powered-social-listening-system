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
        setMeta(mr);
        setStart(mr?.min || "");
        setEnd(mr?.max || "");
        console.log('Date range set:', mr?.min, 'to', mr?.max);
      } catch (error) {
        console.error("Failed to load metadata:", error);
        console.error("Error details:", error.response?.data || error.message);
        // Set fallback data if API fails
        setMeta({ min: "2025-08-04", max: "2025-08-30" });
        setStart("2025-08-04");
        setEnd("2025-08-30");
      } finally {
        setLoading(false);
      }
    };

    // Add timeout to prevent hanging
    const timeoutId = setTimeout(() => {
      console.log('Metadata loading timeout - using fallback');
      setMeta({ min: "2025-08-04", max: "2025-08-30" });
      setStart("2025-08-04");
      setEnd("2025-08-30");
      setLoading(false);
    }, 5000);

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

