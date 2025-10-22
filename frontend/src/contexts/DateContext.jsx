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
    (async () => {
      try {
        const mr = await getMeta();
        setMeta(mr);
        setStart(mr?.min || "");
        setEnd(mr?.max || "");
      } catch (error) {
        console.error("Failed to load metadata:", error);
      } finally {
        setLoading(false);
      }
    })();
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

