#!/bin/bash

# Standalone FastAPI Backend Startup Script
# This script starts only the FastAPI backend server for heart rate detection

echo "🚀 Starting Heart Rate Detection Backend API..."
echo "============================================"

# Check if we're in the right directory
if [ ! -f "server.py" ]; then
    echo "❌ Error: Please run this script from the facial-heart-rate directory"
    echo "   The directory should contain server.py file"
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down server..."
    
    # Kill FastAPI server
    if [ ! -z "$FASTAPI_PID" ]; then
        kill $FASTAPI_PID 2>/dev/null
        echo "   ✅ FastAPI server stopped"
    fi
    
    echo "👋 Goodbye!"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "🎯 Heart Rate Detection Backend API v2.0.0"
echo "   📹 Live Camera Feed Processing"
echo "   📊 Real-time Heart Rate Analysis"
echo "   🔄 RESTful API Endpoints"
echo "   📋 OpenAPI Documentation"
echo ""

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

# Check if required packages are installed
echo "🔧 Checking dependencies..."
python -c "import fastapi, uvicorn, cv2, mediapipe, numpy, scipy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing requirements..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "   ✅ Dependencies verified"

# Check camera availability (basic check)
echo "🔧 Checking camera access..."
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Camera check failed. Ensure camera is connected and accessible."
fi

# Start FastAPI server
echo "🔧 Starting FastAPI backend server..."
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload &
FASTAPI_PID=$!

# Wait a moment for FastAPI to start
echo "   ⏳ Waiting for server initialization..."
sleep 3

# Check if FastAPI started successfully
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ Failed to start FastAPI server"
    cleanup
    exit 1
fi

echo "   ✅ FastAPI server running on http://localhost:8000"
echo ""
echo "🎉 Backend API is ready!"
echo ""
echo "📋 Available Endpoints:"
echo "   🏠 Health Check:     http://localhost:8000/health"
echo "   ❤️  Heart Rate:      http://localhost:8000/bpm"
echo "   📊 System Status:    http://localhost:8000/status"
echo "   📹 Video Feed:       http://localhost:8000/video_feed"
echo "   🖼️  Current Frame:    http://localhost:8000/current_frame"
echo "   📊 Metrics:          http://localhost:8000/metrics"
echo ""
echo "📚 API Documentation:"
echo "   📋 Swagger UI:       http://localhost:8000/docs"
echo "   📖 ReDoc:            http://localhost:8000/redoc"
echo ""
echo "🔧 Integration Examples:"
echo "   Frontend Setup:      Create Next.js app with Tailwind CSS"
echo "   API Base URL:        http://localhost:8000"
echo "   WebSocket Support:   Available for real-time updates"
echo ""
echo "⚡ Press Ctrl+C to stop the server"
echo ""

# Wait for user interrupt
wait
