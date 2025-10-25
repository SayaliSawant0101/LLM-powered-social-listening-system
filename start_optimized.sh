#!/bin/bash
# Walmart Social Listener - Optimized Startup Script

echo "🚀 Starting Walmart Social Listener with optimizations..."

# Load configuration
if [ -f "config.env" ]; then
    echo "📋 Loading configuration from config.env"
    source config.env
fi

# Set default optimizations
export THEMES_FAST_MODE=${THEMES_FAST_MODE:-true}
export THEMES_EMB_BACKEND=${THEMES_EMB_BACKEND:-tfidf}

echo "⚡ Fast Mode: $THEMES_FAST_MODE"
echo "🔧 Embedding Backend: $THEMES_EMB_BACKEND"

# Start FastAPI backend
echo "🐍 Starting FastAPI backend..."
cd "/Users/sayalisawant/Projects/walmart_social_listener copy"
source .venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Start Node.js server
echo "🟢 Starting Node.js server..."
cd "/Users/sayalisawant/Projects/walmart_social_listener copy/server"
npm start &
NODE_PID=$!

# Start React frontend
echo "⚛️ Starting React frontend..."
cd "/Users/sayalisawant/Projects/walmart_social_listener copy/frontend"
npm run dev &
REACT_PID=$!

echo ""
echo "✅ All servers started!"
echo "📊 FastAPI Backend: http://localhost:8000"
echo "🔄 Node.js Server: http://localhost:3001"
echo "⚛️ React Frontend: http://localhost:5173 (or 5174)"
echo ""
echo "🎯 Theme generation is now optimized for speed!"
echo "💡 Use THEMES_FAST_MODE=true for ultra-fast processing (no LLM calls)"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user interrupt
trap 'echo "🛑 Stopping servers..."; kill $FASTAPI_PID $NODE_PID $REACT_PID; exit' INT
wait
