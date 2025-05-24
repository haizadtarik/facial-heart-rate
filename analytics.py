"""
Advanced analytics dashboard for heart rate detection system.
Provides detailed historical analysis and insights.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import requests
from data_logger import HeartRateLogger

def create_analytics_dashboard():
    """Create an advanced analytics dashboard."""
    st.set_page_config(
        page_title="Heart Rate Analytics",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Heart Rate Analytics Dashboard")
    
    # Initialize logger
    logger = HeartRateLogger()
    
    # Sidebar controls
    st.sidebar.header("Analysis Options")
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"],
        index=2
    )
    
    # Convert time range to hours
    time_mapping = {
        "Last Hour": 1,
        "Last 6 Hours": 6,
        "Last 24 Hours": 24,
        "Last Week": 168
    }
    hours = time_mapping[time_range]
    
    # Get data
    stats = logger.get_session_stats(hours)
    readings = logger.get_recent_readings(1000)
    
    if not readings:
        st.warning("No data available. Start the heart rate detection system to collect data.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(readings)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter by time range
    cutoff_time = datetime.now() - timedelta(hours=hours)
    df_filtered = df[df['timestamp'] > cutoff_time]
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average BPM",
            f"{stats.get('avg_bpm', 0):.1f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Min BPM",
            stats.get('min_bpm', 0)
        )
    
    with col3:
        st.metric(
            "Max BPM",
            stats.get('max_bpm', 0)
        )
    
    with col4:
        st.metric(
            "Total Readings",
            stats.get('total_readings', 0)
        )
    
    # Charts
    if not df_filtered.empty:
        # Time series chart
        st.subheader("Heart Rate Over Time")
        fig_time = go.Figure()
        
        # Add BPM line
        fig_time.add_trace(go.Scatter(
            x=df_filtered['timestamp'],
            y=df_filtered['bpm'],
            mode='lines+markers',
            name='Heart Rate',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))
        
        # Add normal range bands
        fig_time.add_hline(y=60, line_dash="dash", line_color="green", 
                          annotation_text="Normal Lower (60 BPM)")
        fig_time.add_hline(y=100, line_dash="dash", line_color="green", 
                          annotation_text="Normal Upper (100 BPM)")
        
        fig_time.update_layout(
            title=f"Heart Rate Trends - {time_range}",
            xaxis_title="Time",
            yaxis_title="BPM",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("BPM Distribution")
            fig_hist = px.histogram(
                df_filtered, 
                x='bpm', 
                nbins=20,
                title="Heart Rate Distribution",
                color_discrete_sequence=['red']
            )
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("Confidence Levels")
            confidence_counts = df_filtered['confidence'].value_counts()
            fig_pie = px.pie(
                values=confidence_counts.values,
                names=confidence_counts.index,
                title="Reading Confidence Distribution"
            )
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Recent readings table
        st.subheader("Recent Readings")
        display_df = df_filtered.tail(20).copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
        st.dataframe(
            display_df[['timestamp', 'bpm', 'confidence', 'signal_quality']],
            use_container_width=True
        )
        
        # Export functionality
        st.subheader("Data Export")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Current Data"):
                filename = f"heart_rate_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                logger.export_data(filename, hours)
                st.success(f"Data exported to {filename}")
        
        with col2:
            if st.button("Clear Old Data (>7 days)"):
                # This would require a method in the logger to clean old data
                st.info("Feature coming soon")
    
    else:
        st.info(f"No data available for the selected time range ({time_range})")

if __name__ == "__main__":
    create_analytics_dashboard()
