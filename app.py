"""
Enhanced Streamlit application with multi-page support for heart rate detection.
Includes real-time monitoring, analytics, and health alerts.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh

# Import our new modules
try:
    from data_logger import HeartRateLogger
    from health_monitor import health_monitor, print_alert_callback, AlertLevel
    DATA_LOGGING_AVAILABLE = True
except ImportError:
    DATA_LOGGING_AVAILABLE = False
    st.warning("Advanced features not available. Install additional dependencies.")

# Page configuration
st.set_page_config(
    page_title="Heart Rate Monitor Pro",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'logger' not in st.session_state and DATA_LOGGING_AVAILABLE:
    st.session_state.logger = HeartRateLogger()
    health_monitor.add_alert_callback(print_alert_callback)

# Sidebar navigation
st.sidebar.title("❤️ Heart Rate Monitor Pro")
page = st.sidebar.selectbox(
    "Select Page",
    ["🔴 Live Monitor", "📊 Analytics", "⚠️ Health Alerts", "⚙️ Settings"]
)

# API Base URL
API_BASE = "http://localhost:8000"

def check_api_connection():
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_BASE}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_current_reading():
    """Get current heart rate reading from API."""
    try:
        response = requests.get(f"{API_BASE}/bpm", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_system_status():
    """Get system status from API."""
    try:
        response = requests.get(f"{API_BASE}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_current_frame():
    """Get current camera frame as base64 image."""
    try:
        response = requests.get(f"{API_BASE}/current_frame", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# Main content based on selected page
if page == "🔴 Live Monitor":
    st.title("🔴 Live Heart Rate Monitor")
    
    # Check API connection
    if not check_api_connection():
        st.error("❌ Cannot connect to heart rate detection server!")
        st.info("Make sure the FastAPI server is running on port 8000")
        st.code("python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload")
        st.stop()
    
    # Auto-refresh every 2 seconds
    count = st_autorefresh(interval=2000, limit=None, key="live_monitor")
    
    # Get current data
    reading = get_current_reading()
    status = get_system_status()
    
    if reading and status:
        # Log the reading if available
        if DATA_LOGGING_AVAILABLE and 'logger' in st.session_state:
            st.session_state.logger.log_reading(
                reading['bpm'],
                reading['confidence'],
                reading['face_detected'],
                reading['buffer_fill'],
                status.get('signal_quality', 'unknown')
            )
            
            # Process reading for health monitoring
            health_monitor.process_reading(reading)
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # BPM with color coding
            bpm = reading['bpm']
            if bpm < 60:
                color = "🔵"
            elif bpm > 100:
                color = "🔴"
            else:
                color = "🟢"
            
            st.metric(
                f"{color} Heart Rate",
                f"{bpm} BPM",
                delta=None
            )
        
        with col2:
            confidence_color = "🟢" if reading['confidence'] == "high" else "🟡"
            st.metric(
                f"{confidence_color} Confidence",
                reading['confidence'].title()
            )
        
        with col3:
            face_status = "🟢 Detected" if reading['face_detected'] else "🔴 Not Found"
            st.metric("Face Detection", face_status)
        
        with col4:
            signal_color = {"good": "🟢", "fair": "🟡", "poor": "🔴"}.get(status.get('signal_quality', 'unknown'), "❓")
            st.metric(
                f"{signal_color} Signal Quality", 
                status.get('signal_quality', 'Unknown').title()
            )
        
        # Live Camera Feed and Chart Section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📹 Live Camera Feed")
            
            # Get current frame
            frame_data = get_current_frame()
            if frame_data and frame_data.get('frame'):
                # Display the camera feed with overlay
                st.image(
                    f"data:image/jpeg;base64,{frame_data['frame']}", 
                    caption=f"Face Detection: {'✅ Active' if frame_data.get('face_detected') else '❌ Inactive'} | BPM: {frame_data.get('bpm', 0)}",
                    use_column_width=True
                )
            else:
                st.warning("📹 Camera feed not available")
                st.info("Make sure the camera is connected and the backend server is running")
        
        with col2:
            # Real-time chart
            if DATA_LOGGING_AVAILABLE and 'logger' in st.session_state:
                st.subheader("📈 Real-time Trend (Last 5 Minutes)")
                
                recent_readings = st.session_state.logger.get_recent_readings(150)  # ~5 minutes at 2s intervals
                
                if recent_readings:
                    df = pd.DataFrame(recent_readings)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Filter last 5 minutes
                    cutoff_time = datetime.now() - timedelta(minutes=5)
                    df_recent = df[df['timestamp'] > cutoff_time]
                    
                    if not df_recent.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_recent['timestamp'],
                            y=df_recent['bpm'],
                            mode='lines+markers',
                            name='Heart Rate',
                            line=dict(color='red', width=3),
                            marker=dict(size=6)
                        ))
                        
                        # Add normal range
                        fig.add_hline(y=60, line_dash="dash", line_color="green", opacity=0.7)
                        fig.add_hline(y=100, line_dash="dash", line_color="green", opacity=0.7)
                        
                        fig.update_layout(
                            title="Heart Rate Trend",
                            xaxis_title="Time",
                            yaxis_title="BPM",
                            height=350,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No recent data for trend chart")
            else:
                st.subheader("📊 Heart Rate Info")
                st.metric("Current BPM", f"{reading['bpm']}")
                st.metric("Buffer Status", reading['buffer_fill'])
                st.metric("Confidence", reading['confidence'].title())
        
        
        # System status details
        st.subheader("🖥️ System Status")
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            st.info(f"📹 Camera: {'Active' if status['camera_active'] else 'Inactive'}")
            st.info(f"📊 Buffer: {status['buffer_size']}/{status['max_buffer_size']}")
        
        with status_col2:
            st.info(f"⏱️ Last Update: {status['time_since_last_update']:.2f}s ago")
            st.info(f"📡 Signal: {status['signal_quality'].title()}")
        
        # Instructions
        if not reading['face_detected']:
            st.warning("⚠️ Position your face in front of the camera for heart rate detection")
    
    else:
        st.error("❌ Unable to get heart rate data")

elif page == "📊 Analytics" and DATA_LOGGING_AVAILABLE:
    st.title("📊 Heart Rate Analytics")
    
    logger = st.session_state.logger
    
    # Time range selector
    time_range = st.selectbox(
        "Select Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"],
        index=2
    )
    
    time_mapping = {
        "Last Hour": 1,
        "Last 6 Hours": 6, 
        "Last 24 Hours": 24,
        "Last Week": 168
    }
    hours = time_mapping[time_range]
    
    # Get statistics
    stats = logger.get_session_stats(hours)
    readings = logger.get_recent_readings(2000)
    
    if stats and readings:
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average BPM", f"{stats['avg_bpm']:.1f}")
        with col2:
            st.metric("Min BPM", stats['min_bpm'])
        with col3:
            st.metric("Max BPM", stats['max_bpm'])
        with col4:
            st.metric("Total Readings", stats['total_readings'])
        
        # Charts
        df = pd.DataFrame(readings)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by time range
        cutoff_time = datetime.now() - timedelta(hours=hours)
        df_filtered = df[df['timestamp'] > cutoff_time]
        
        if not df_filtered.empty:
            # Time series
            st.subheader("Heart Rate Over Time")
            fig_time = px.line(
                df_filtered, 
                x='timestamp', 
                y='bpm',
                title=f"Heart Rate Trends - {time_range}"
            )
            fig_time.add_hline(y=60, line_dash="dash", line_color="green")
            fig_time.add_hline(y=100, line_dash="dash", line_color="green")
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Distribution charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("BPM Distribution")
                fig_hist = px.histogram(df_filtered, x='bpm', nbins=20)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.subheader("Confidence Levels")
                confidence_counts = df_filtered['confidence'].value_counts()
                fig_pie = px.pie(values=confidence_counts.values, names=confidence_counts.index)
                st.plotly_chart(fig_pie, use_container_width=True)
    
    else:
        st.info("No data available for the selected time range")

elif page == "⚠️ Health Alerts" and DATA_LOGGING_AVAILABLE:
    st.title("⚠️ Health Monitoring & Alerts")
    
    # Alert summary
    alert_summary = health_monitor.get_alert_summary()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 Critical Alerts", alert_summary['critical_alerts'])
    with col2:
        st.metric("🟡 Warning Alerts", alert_summary['warning_alerts'])
    with col3:
        st.metric("🔵 Info Alerts", alert_summary['info_alerts'])
    
    # Recent alerts
    st.subheader("Recent Alerts (Last 30 Minutes)")
    recent_alerts = health_monitor.get_recent_alerts(30)
    
    if recent_alerts:
        for alert in reversed(recent_alerts[-10:]):  # Show last 10
            level_color = {
                AlertLevel.CRITICAL: "🔴",
                AlertLevel.WARNING: "🟡", 
                AlertLevel.INFO: "🔵"
            }
            
            color = level_color.get(alert.level, "❓")
            timestamp = alert.timestamp.strftime("%H:%M:%S")
            
            st.write(f"{color} **{timestamp}** - {alert.message}")
    else:
        st.success("✅ No recent alerts")
    
    # Alert settings
    st.subheader("Alert Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        high_bpm_threshold = st.slider(
            "High BPM Alert Threshold", 
            100, 180, 
            health_monitor.thresholds['high_bpm']
        )
        health_monitor.thresholds['high_bpm'] = high_bpm_threshold
    
    with col2:
        low_bpm_threshold = st.slider(
            "Low BPM Alert Threshold", 
            30, 80, 
            health_monitor.thresholds['low_bpm']
        )
        health_monitor.thresholds['low_bpm'] = low_bpm_threshold

elif page == "⚙️ Settings":
    st.title("⚙️ System Settings")
    
    # API Settings
    st.subheader("API Connection")
    api_status = "🟢 Connected" if check_api_connection() else "🔴 Disconnected"
    st.write(f"Status: {api_status}")
    st.write(f"Endpoint: {API_BASE}")
    
    # Data Management
    if DATA_LOGGING_AVAILABLE:
        st.subheader("Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Data"):
                filename = f"heart_rate_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.session_state.logger.export_data(filename, 24)
                st.success(f"Data exported to {filename}")
        
        with col2:
            if st.button("View Database Info"):
                readings = st.session_state.logger.get_recent_readings(1)
                if readings:
                    st.info(f"Database contains data since {readings[0]['timestamp']}")
                else:
                    st.info("No data in database")
    
    # System Information
    st.subheader("System Information")
    st.info(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.info(f"Data Logging: {'✅ Enabled' if DATA_LOGGING_AVAILABLE else '❌ Disabled'}")
    
    # Instructions
    st.subheader("Quick Start")
    st.markdown("""
    1. **Start Backend**: `python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload`
    2. **Start Frontend**: `streamlit run app_enhanced.py --server.port 8501`
    3. **Position yourself** in front of the camera
    4. **Monitor** your heart rate in real-time
    """)

else:
    if not DATA_LOGGING_AVAILABLE:
        st.title("❤️ Heart Rate Monitor")
        st.warning("⚠️ Advanced features are not available")
        st.info("Install additional dependencies to enable analytics and health monitoring")
        st.code("pip install pandas")
        
        # Basic monitoring fallback
        if check_api_connection():
            reading = get_current_reading()
            if reading:
                st.metric("Heart Rate", f"{reading['bpm']} BPM")
                st.metric("Confidence", reading['confidence'].title())
                face_status = "Detected" if reading['face_detected'] else "Not Found"
                st.metric("Face Detection", face_status)
