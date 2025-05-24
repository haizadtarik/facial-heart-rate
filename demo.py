#!/usr/bin/env python3
"""
Demo script for the Enhanced Heart Rate Detection System.
Demonstrates all the new features and capabilities.
"""

import requests
import time
import json
from datetime import datetime

API_BASE = "http://localhost:8000"

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def check_system_status():
    """Check if the system is running and get status."""
    print_header("System Status Check")
    
    try:
        # Check API connection
        response = requests.get(f"{API_BASE}/", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI Backend: Connected")
        else:
            print("❌ FastAPI Backend: Error")
            return False
            
        # Get detailed status
        status_response = requests.get(f"{API_BASE}/status", timeout=5)
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"📹 Camera: {'Active' if status['camera_active'] else 'Inactive'}")
            print(f"👤 Face Detection: {'Active' if status['face_detected'] else 'Inactive'}")
            print(f"📊 Buffer: {status['buffer_size']}/{status['max_buffer_size']}")
            print(f"📡 Signal Quality: {status['signal_quality'].title()}")
            print(f"⏱️  Last Update: {status['time_since_last_update']:.2f}s ago")
            return True
        else:
            print("❌ Status endpoint: Error")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return False

def demo_real_time_monitoring():
    """Demonstrate real-time heart rate monitoring."""
    print_header("Real-time Heart Rate Monitoring")
    
    print("📊 Collecting 10 real-time readings...")
    readings = []
    
    for i in range(10):
        try:
            response = requests.get(f"{API_BASE}/bpm", timeout=5)
            if response.status_code == 200:
                reading = response.json()
                readings.append(reading)
                
                # Display reading
                bpm = reading['bpm']
                confidence = reading['confidence']
                face_detected = "👤" if reading['face_detected'] else "❌"
                
                print(f"Reading {i+1:2d}: {bpm:3d} BPM | {confidence:>6s} | {face_detected}")
                
            time.sleep(2)  # Wait 2 seconds between readings
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Reading {i+1} failed: {e}")
    
    if readings:
        # Calculate statistics
        bpms = [r['bpm'] for r in readings]
        avg_bpm = sum(bpms) / len(bpms)
        min_bpm = min(bpms)
        max_bpm = max(bpms)
        high_confidence = len([r for r in readings if r['confidence'] == 'high'])
        
        print(f"\n📈 Session Statistics:")
        print(f"   Average BPM: {avg_bpm:.1f}")
        print(f"   Range: {min_bpm} - {max_bpm} BPM")
        print(f"   High Confidence: {high_confidence}/{len(readings)}")

def demo_data_logging():
    """Demonstrate data logging capabilities."""
    print_header("Data Logging Demo")
    
    try:
        from data_logger import HeartRateLogger
        
        print("💾 Initializing data logger...")
        logger = HeartRateLogger()
        
        # Get some readings to log
        print("📝 Logging current readings...")
        for i in range(5):
            response = requests.get(f"{API_BASE}/bpm", timeout=5)
            if response.status_code == 200:
                reading = response.json()
                
                # Log the reading
                logger.log_reading(
                    reading['bpm'],
                    reading['confidence'],
                    reading['face_detected'],
                    reading['buffer_fill'],
                    'good'  # Assume good signal quality for demo
                )
                
                print(f"   Logged: {reading['bpm']} BPM ({reading['confidence']})")
                time.sleep(1)
        
        # Get statistics
        print("\n📊 Database Statistics:")
        stats = logger.get_session_stats(1)  # Last hour
        if stats:
            print(f"   Total Readings: {stats['total_readings']}")
            print(f"   Average BPM: {stats['avg_bpm']:.1f}")
            print(f"   Range: {stats['min_bpm']} - {stats['max_bpm']} BPM")
            print(f"   High Confidence: {stats['high_confidence_readings']}")
        
        # Get recent readings
        recent = logger.get_recent_readings(5)
        if recent:
            print(f"\n🕒 Recent Readings:")
            for reading in recent:
                timestamp = datetime.fromisoformat(reading['timestamp'].replace('Z', '+00:00'))
                print(f"   {timestamp.strftime('%H:%M:%S')}: {reading['bpm']} BPM")
        
        # Export demo
        export_file = f"demo_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        logger.export_data(export_file, 1)
        print(f"💾 Data exported to: {export_file}")
        
    except ImportError:
        print("❌ Data logging not available (missing dependencies)")
    except Exception as e:
        print(f"❌ Data logging error: {e}")

def demo_health_monitoring():
    """Demonstrate health monitoring and alerts."""
    print_header("Health Monitoring Demo")
    
    try:
        from health_monitor import HealthMonitor, AlertLevel, print_alert_callback
        
        print("⚠️  Initializing health monitor...")
        monitor = HealthMonitor()
        monitor.add_alert_callback(print_alert_callback)
        
        # Set demo thresholds (more sensitive for demo)
        monitor.thresholds['high_bpm'] = 80  # Lower threshold for demo
        monitor.thresholds['low_bpm'] = 60   # Higher threshold for demo
        
        print(f"🎯 Alert thresholds set:")
        print(f"   High BPM: >{monitor.thresholds['high_bpm']} BPM")
        print(f"   Low BPM: <{monitor.thresholds['low_bpm']} BPM")
        
        # Process some readings to trigger alerts
        print("\n🔍 Processing readings for alert demonstration...")
        
        for i in range(5):
            response = requests.get(f"{API_BASE}/bpm", timeout=5)
            if response.status_code == 200:
                reading = response.json()
                monitor.process_reading(reading)
                
                print(f"   Processed: {reading['bpm']} BPM")
                time.sleep(1)
        
        # Show alert summary
        summary = monitor.get_alert_summary()
        print(f"\n📊 Alert Summary:")
        print(f"   Total Alerts: {summary['total_alerts']}")
        print(f"   Critical: {summary['critical_alerts']}")
        print(f"   Warning: {summary['warning_alerts']}")
        print(f"   Info: {summary['info_alerts']}")
        
        # Show recent alerts
        recent_alerts = monitor.get_recent_alerts(10)
        if recent_alerts:
            print(f"\n🚨 Recent Alerts:")
            for alert in recent_alerts[-3:]:  # Show last 3
                timestamp = alert.timestamp.strftime('%H:%M:%S')
                level = alert.level.value.upper()
                print(f"   [{timestamp}] {level}: {alert.message}")
        else:
            print("\n✅ No alerts generated")
        
    except ImportError:
        print("❌ Health monitoring not available (missing dependencies)")
    except Exception as e:
        print(f"❌ Health monitoring error: {e}")

def demo_api_endpoints():
    """Demonstrate all available API endpoints."""
    print_header("API Endpoints Demo")
    
    endpoints = [
        ("/", "Root endpoint"),
        ("/bpm", "Current heart rate"),
        ("/status", "System status")
    ]
    
    for endpoint, description in endpoints:
        try:
            print(f"\n🔗 Testing {endpoint} - {description}")
            response = requests.get(f"{API_BASE}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Status: {response.status_code}")
                print(f"   📄 Response: {json.dumps(data, indent=4)}")
            else:
                print(f"   ❌ Status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error: {e}")

def main():
    """Run the complete demonstration."""
    print("🚀 Enhanced Heart Rate Detection System Demo")
    print("=" * 60)
    print("This demo showcases all the enhanced features of the system.")
    print("Make sure both the FastAPI backend and camera are active.")
    
    # Run demonstrations
    if check_system_status():
        demo_real_time_monitoring()
        demo_data_logging()
        demo_health_monitoring()
        demo_api_endpoints()
        
        print_header("Demo Complete")
        print("🎉 All features demonstrated successfully!")
        print("📱 Open the Streamlit interface to explore interactively:")
        print("   Basic: http://localhost:8501")
        print("   Enhanced: http://localhost:8502")
        
    else:
        print("\n❌ System not ready. Please start the servers first:")
        print("   ./start_enhanced.sh")

if __name__ == "__main__":
    main()
