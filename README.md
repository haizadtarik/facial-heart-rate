# ❤️ Facial Heart Rate Detection System Pro

A comprehensive real-time heart rate monitoring system that uses facial detection and photoplethysmography (PPG) to estimate heart rate from webcam video feed. The system features both basic and enhanced modes with advanced analytics, health monitoring, and data logging capabilities.

## 🌟 Features

### Core Features
- **Real-time heart rate detection** using facial ROI analysis
- **Interactive web interface** with live BPM updates
- **Heart rate trend visualization** with historical data
- **Confidence scoring** for measurement reliability
- **Status monitoring** for camera and face detection
- **Responsive design** with health indicators

### Enhanced Features (Pro Version)
- **📊 Advanced Analytics Dashboard** - Historical data analysis with trends and statistics
- **⚠️ Health Monitoring & Alerts** - Smart alerts for abnormal readings with customizable thresholds
- **💾 Data Logging** - Persistent storage in SQLite database for long-term tracking
- **📈 Real-time Charts** - Interactive Plotly visualizations with 5-minute rolling trends
- **📋 Export Functionality** - Export data to JSON for external analysis
- **🎯 Multi-page Interface** - Organized navigation between monitoring, analytics, and settings

## 🛠️ Technology Stack

- **Backend**: FastAPI, OpenCV, MediaPipe, SciPy
- **Frontend**: Streamlit, Plotly, Streamlit-AutoRefresh
- **Signal Processing**: Bandpass filtering, Power Spectral Density analysis
- **Computer Vision**: MediaPipe Face Mesh for landmark detection
- **Data Storage**: SQLite database for historical data
- **Analytics**: Pandas for data analysis and trend calculation

## 📋 Requirements

- Python 3.8+
- Webcam/Camera
- Good lighting conditions
- macOS, Windows, or Linux

## 🚀 Quick Start (Updated)

```bash
# Start the system (enhanced dashboard only)
./start.sh
```
- Open your browser to http://localhost:8501
- The dashboard will show the live camera feed, heart rate, analytics, and health alerts.

## 🆕 Live Camera Feed

- The enhanced dashboard now includes a real-time live camera feed with face detection and BPM overlay.
- Access the dashboard at: http://localhost:8501
- The basic interface has been removed; all features are now in the enhanced dashboard.

### How to Use the Live Camera Feed
- The camera feed appears on the main dashboard page.
- Face detection status and current BPM are overlaid on the video.
- If no face is detected, a warning is shown on the video feed.

## 📱 Usage Instructions

1. **Position yourself** 2-3 feet away from your camera
2. **Ensure good lighting** illuminates your face evenly
3. **Look directly at the camera** and stay as still as possible
4. **Wait 10-15 seconds** for the system to collect sufficient data
5. **Monitor the confidence level** - green indicates reliable readings

### 💡 Tips for Better Accuracy

- Remove glasses if possible (they can interfere with facial detection)
- Avoid excessive head movement
- Ensure stable, bright lighting (avoid backlighting)
- Keep your entire face visible in the camera frame
- Be patient - the system needs time to collect stable readings

## 🏗️ System Architecture

```
┌─────────────────┐    HTTP Requests    ┌─────────────────┐
│   Streamlit     │ ──────────────────► │   FastAPI       │
│   Frontend      │                     │   Backend       │
│   (Port 8501)   │ ◄────────────────── │   (Port 8000)   │
└─────────────────┘    JSON Responses   └─────────────────┘
                                                │
                                                ▼
                                        ┌─────────────────┐
                                        │   Camera        │
                                        │   Processing    │
                                        │   Thread        │
                                        └─────────────────┘
```

### Backend Components

- **Video Capture Loop**: Continuously processes camera frames
- **Face Detection**: Uses MediaPipe Face Mesh for landmark detection
- **Signal Extraction**: Extracts green channel data from forehead ROI
- **Signal Processing**: Applies bandpass filtering and spectral analysis
- **BPM Calculation**: Uses power spectral density to find heart rate frequency

### Frontend Components

- **Real-time Display**: Shows current BPM with color-coded confidence
- **Status Monitoring**: Displays camera, face detection, and signal quality
- **Trend Visualization**: Interactive chart showing heart rate over time
- **Statistics Panel**: Shows average, min, max BPM values

## 🔧 API Endpoints

### Backend (FastAPI)

- `GET /` - Health check
- `GET /status` - System status (camera, face detection, signal quality)
- `GET /bpm` - Current heart rate and confidence data

### Example Response

```json
{
  "bpm": 72,
  "confidence": "high",
  "message": "Face detected",
  "face_detected": true,
  "buffer_fill": "180/200",
  "timestamp": 1716569234.123
}
```

## ⚙️ Configuration

Key parameters in `server.py`:

```python
BUFFER_SIZE = 200          # Signal buffer size (~10s at 20 FPS)
MIN_HZ, MAX_HZ = 0.7, 4.0  # BPM range: 42-240
SAMPLING_RATE = 20         # Target frame processing rate
CONFIDENCE_THRESHOLD = 0.6 # Face detection confidence
```

## 🧪 Testing

Test the backend directly:

```bash
# Check server status
curl http://localhost:8000/status

# Get current BPM
curl http://localhost:8000/bpm
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera not accessible**
   - Check camera permissions
   - Ensure no other applications are using the camera
   - Try restarting the backend server

2. **No face detected**
   - Improve lighting conditions
   - Move closer to camera (2-3 feet optimal)
   - Ensure face is fully visible
   - Remove glasses or face coverings

3. **Unstable BPM readings**
   - Stay still during measurement
   - Ensure consistent lighting
   - Wait for buffer to fill (watch buffer status)
   - Check confidence level (should be "high")

4. **High CPU usage**
   - Reduce camera resolution in `server.py`
   - Increase processing interval
   - Close unnecessary applications

### Error Messages

- `"Camera not active"` - Backend cannot access camera
- `"No recent face detection"` - Face not visible for >5 seconds
- `"Face detection intermittent"` - Unstable face detection

## 📊 Accuracy Notes

This system is designed for **demonstration and educational purposes**. While it can provide reasonable heart rate estimates under optimal conditions, it should not be used for medical diagnosis or monitoring. Factors affecting accuracy:

- Lighting conditions
- Camera quality
- Subject movement
- Skin tone and complexion
- Environmental interference

For medical-grade heart rate monitoring, use dedicated medical devices.

## 🔬 How It Works

The system uses **remote photoplethysmography (rPPG)** to detect minute color changes in facial skin caused by blood volume variations with each heartbeat:

1. **Face Detection**: MediaPipe identifies facial landmarks
2. **ROI Extraction**: Forehead region selected for signal extraction
3. **Color Analysis**: Green channel monitored for blood volume changes
4. **Signal Processing**: Bandpass filtering removes noise and artifacts
5. **Frequency Analysis**: Power spectral density identifies heart rate frequency
6. **BPM Calculation**: Peak frequency converted to beats per minute

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**⚠️ Disclaimer**: This software is for educational and demonstration purposes only. It is not intended for medical use or diagnosis. Always consult healthcare professionals for medical advice.

## 🎯 Enhanced Features Guide

### 📊 Analytics Dashboard

The analytics dashboard provides comprehensive insights into your heart rate data:

**Features:**
- **Time Range Selection**: View data from last hour to last week
- **Statistical Summary**: Average, min, max BPM with total readings count
- **Trend Visualization**: Interactive time-series charts with normal range indicators
- **Distribution Analysis**: Histogram showing BPM distribution patterns
- **Confidence Metrics**: Pie chart showing reliability of readings

**Navigation:**
1. Select "📊 Analytics" from the sidebar
2. Choose your desired time range
3. Explore the various charts and statistics

### ⚠️ Health Monitoring & Alerts

Real-time health monitoring with customizable alert system:

**Alert Types:**
- **High Heart Rate**: Configurable threshold (default: 120 BPM)
- **Low Heart Rate**: Configurable threshold (default: 50 BPM)  
- **System Errors**: API connection issues, camera problems
- **Signal Quality**: Poor detection or face visibility issues

**Features:**
- **Real-time Monitoring**: Continuous background monitoring
- **Alert History**: Last 30 minutes of alerts with timestamps
- **Customizable Thresholds**: Adjust alert levels in settings
- **Alert Levels**: Critical (🔴), Warning (🟡), Info (🔵)

### 💾 Data Logging

Persistent data storage for long-term tracking:

**Database Features:**
- **SQLite Storage**: Local database for readings and sessions
- **Automatic Logging**: All readings saved with timestamps
- **Session Statistics**: Calculate trends over time periods
- **Data Export**: JSON export for external analysis

**Data Retention:**
- Readings stored with timestamp, BPM, confidence, and signal quality
- Historical statistics calculated for any time period
- Export functionality for data backup and analysis

### 📱 Application Modes

**Basic Mode** (`app.py`):
- Real-time monitoring
- Simple dashboard with current readings
- Status indicators
- Basic trend visualization

**Enhanced Mode** (`app_enhanced.py`):
- All basic features plus:
- Multi-page navigation
- Advanced analytics dashboard  
- Health monitoring with alerts
- Data logging and export
- Comprehensive settings panel

### 🔧 Configuration Options

**Health Alert Thresholds:**
```python
# Default thresholds (can be adjusted in app)
high_bpm_threshold = 120    # BPM above this triggers warning
low_bpm_threshold = 50      # BPM below this triggers warning
poor_signal_duration = 30   # Seconds of poor signal before alert
no_face_duration = 10       # Seconds without face before alert
```

**Data Storage:**
```python
# Database location
database_path = "heart_rate_data.db"

# Data retention
max_alerts = 100           # Maximum alerts kept in memory
export_format = "JSON"     # Export file format
```

## 🧪 Testing & Validation

### System Test Script

Run the included test script to validate your setup:

```bash
python test_system.py
```

The test script verifies:
- API connectivity
- Camera initialization
- Face detection capability
- Heart rate calculation
- Data logging (if available)

### Performance Monitoring

Monitor system performance through:
- **Backend Logs**: Check FastAPI console for processing times
- **Frontend Status**: View connection status in Streamlit sidebar
- **Database Size**: Monitor growth of heart_rate_data.db
- **CPU Usage**: Watch system resources during operation

### Quality Assurance

For optimal results:
1. **Lighting**: Use consistent, diffused lighting
2. **Position**: Maintain 2-3 feet distance from camera
3. **Stability**: Minimize movement during readings
4. **Environment**: Reduce background motion and shadows
5. **Calibration**: Allow 30-60 seconds for system stabilization

## 📈 Data Analysis Tips

### Understanding Your Data

**Normal Ranges:**
- Resting: 60-100 BPM
- Light Activity: 100-120 BPM
- Exercise: 120-160 BPM

**Confidence Levels:**
- **High**: Stable face detection, good signal quality
- **Medium**: Intermittent detection or moderate signal
- **Low**: Poor detection or unstable signal

**Trend Analysis:**
- Look for patterns over time
- Monitor changes with activity
- Track improvements in signal quality
- Compare different lighting conditions

### Export and External Analysis

Export your data for advanced analysis:

```bash
# Data exported to JSON includes:
{
  "export_time": "2025-05-24T22:50:00",
  "statistics": {
    "avg_bpm": 72.5,
    "min_bpm": 65,
    "max_bpm": 95,
    "total_readings": 150
  },
  "readings": [
    {
      "timestamp": "2025-05-24T22:49:30",
      "bpm": 73,
      "confidence": "high",
      "face_detected": true,
      "signal_quality": "good"
    }
    // ... more readings
  ]
}
```

## 🎛️ API Endpoints (Advanced)

For developers wanting to integrate with the system:

```bash
# Health check
GET http://localhost:8000/

# Current heart rate
GET http://localhost:8000/bpm

# System status  
GET http://localhost:8000/status

# API documentation
GET http://localhost:8000/docs
```