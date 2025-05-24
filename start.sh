#!/bin/bash

# Enhanced Heart Rate Detection System Startup Script
# This script starts both the FastAPI backend and Streamlit frontend

echo "🚀 Starting Enhanced Heart Rate Detection System..."
echo "============================================"

# Check if we're in the right directory
if [ ! -f "server.py" ] || [ ! -f "app.py" ]; then
    echo "❌ Error: Please run this script from the facial-heart-rate directory"
    echo "   The directory should contain server.py and app.py files"
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    
    # Kill FastAPI server
    if [ ! -z "$FASTAPI_PID" ]; then
        kill $FASTAPI_PID 2>/dev/null
        echo "   ✅ FastAPI server stopped"
    fi
    
    # Kill Streamlit server
    if [ ! -z "$STREAMLIT_PID" ]; then
        kill $STREAMLIT_PID 2>/dev/null
        echo "   ✅ Streamlit server stopped"
    fi
    
    echo "👋 Goodbye!"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "🎯 Enhanced Heart Rate Detection System"
echo "   📹 Live Camera Feed"
echo "   📊 Analytics Dashboard"
echo "   ⚠️  Health Monitoring"
echo "   💾 Data Logging"
echo ""

# Start FastAPI server in background
echo "🔧 Starting FastAPI backend server..."
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload &
FASTAPI_PID=$!

# Wait a moment for FastAPI to start
sleep 3

# Check if FastAPI started successfully
if ! curl -s http://localhost:8000/ > /dev/null; then
    echo "❌ Failed to start FastAPI server"
    cleanup
    exit 1
fi

echo "   ✅ FastAPI server running on http://localhost:8000"

# Start Streamlit frontend in background
echo "🎨 Starting Streamlit frontend..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

# Wait a moment for Streamlit to start
sleep 5

# Check if Streamlit started successfully
if ! curl -s http://localhost:8501/ > /dev/null; then
    echo "❌ Failed to start Streamlit server"
    cleanup
    exit 1
fi

echo "   ✅ Streamlit server running on http://localhost:8501"
echo ""
echo "🎉 System is ready!"
echo "📱 Open your browser to: http://localhost:8501"
echo ""
echo "🛠️  Backend API documentation: http://localhost:8000/docs"
echo "⚡ Press Ctrl+C to stop both servers"
echo ""

# Wait for user interrupt
wait
