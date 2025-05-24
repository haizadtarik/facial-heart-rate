"""
Health monitoring and alert system for heart rate detection.
Provides real-time alerts for abnormal readings and system issues.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertType(Enum):
    HIGH_HEART_RATE = "high_heart_rate"
    LOW_HEART_RATE = "low_heart_rate"
    SYSTEM_ERROR = "system_error"
    CAMERA_DISCONNECTED = "camera_disconnected"
    POOR_SIGNAL_QUALITY = "poor_signal_quality"

@dataclass
class Alert:
    type: AlertType
    level: AlertLevel
    message: str
    timestamp: datetime
    value: Optional[float] = None
    
class HealthMonitor:
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        self.monitoring = False
        self.monitor_thread = None
        
        # Configurable thresholds
        self.thresholds = {
            'high_bpm': 120,
            'low_bpm': 50,
            'max_poor_signal_duration': 30,  # seconds
            'max_no_face_duration': 10,  # seconds
        }
        
        # State tracking
        self.last_reading_time = None
        self.poor_signal_start = None
        self.no_face_start = None
        
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)
    
    def trigger_alert(self, alert_type: AlertType, level: AlertLevel, 
                     message: str, value: Optional[float] = None):
        """Trigger a new alert."""
        alert = Alert(
            type=alert_type,
            level=level,
            message=message,
            timestamp=datetime.now(),
            value=value
        )
        
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")
    
    def check_heart_rate(self, bpm: int, confidence: str):
        """Check if heart rate is within normal ranges."""
        if confidence == "high":  # Only alert on high confidence readings
            if bpm > self.thresholds['high_bpm']:
                self.trigger_alert(
                    AlertType.HIGH_HEART_RATE,
                    AlertLevel.WARNING,
                    f"High heart rate detected: {bpm} BPM",
                    bpm
                )
            elif bpm < self.thresholds['low_bpm']:
                self.trigger_alert(
                    AlertType.LOW_HEART_RATE,
                    AlertLevel.WARNING,
                    f"Low heart rate detected: {bpm} BPM",
                    bpm
                )
    
    def check_signal_quality(self, face_detected: bool, signal_quality: str):
        """Monitor signal quality and face detection."""
        now = datetime.now()
        
        # Check face detection
        if not face_detected:
            if self.no_face_start is None:
                self.no_face_start = now
            elif (now - self.no_face_start).seconds > self.thresholds['max_no_face_duration']:
                self.trigger_alert(
                    AlertType.CAMERA_DISCONNECTED,
                    AlertLevel.WARNING,
                    "No face detected for extended period"
                )
                self.no_face_start = now  # Reset to avoid spam
        else:
            self.no_face_start = None
        
        # Check signal quality
        if signal_quality == "poor":
            if self.poor_signal_start is None:
                self.poor_signal_start = now
            elif (now - self.poor_signal_start).seconds > self.thresholds['max_poor_signal_duration']:
                self.trigger_alert(
                    AlertType.POOR_SIGNAL_QUALITY,
                    AlertLevel.WARNING,
                    "Poor signal quality detected for extended period"
                )
                self.poor_signal_start = now  # Reset to avoid spam
        else:
            self.poor_signal_start = None
    
    def process_reading(self, reading_data: Dict):
        """Process a new reading and check for alerts."""
        self.last_reading_time = datetime.now()
        
        # Extract data
        bpm = reading_data.get('bpm', 0)
        confidence = reading_data.get('confidence', 'low')
        face_detected = reading_data.get('face_detected', False)
        
        # Determine signal quality from various factors
        signal_quality = "good"
        if not face_detected or confidence == "low":
            signal_quality = "poor"
        
        # Run checks
        self.check_heart_rate(bpm, confidence)
        self.check_signal_quality(face_detected, signal_quality)
    
    def get_recent_alerts(self, minutes: int = 30) -> List[Alert]:
        """Get alerts from the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [alert for alert in self.alerts if alert.timestamp > cutoff]
    
    def get_alert_summary(self) -> Dict:
        """Get a summary of recent alerts."""
        recent_alerts = self.get_recent_alerts()
        
        summary = {
            'total_alerts': len(recent_alerts),
            'critical_alerts': len([a for a in recent_alerts if a.level == AlertLevel.CRITICAL]),
            'warning_alerts': len([a for a in recent_alerts if a.level == AlertLevel.WARNING]),
            'info_alerts': len([a for a in recent_alerts if a.level == AlertLevel.INFO]),
            'last_alert': recent_alerts[-1] if recent_alerts else None
        }
        
        return summary
    
    def start_monitoring(self, api_url: str = "http://localhost:8000"):
        """Start continuous monitoring in a background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(api_url,),
            daemon=True
        )
        self.monitor_thread.start()
        
        print("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("Health monitoring stopped")
    
    def _monitor_loop(self, api_url: str):
        """Main monitoring loop."""
        import requests
        
        while self.monitoring:
            try:
                # Get current reading
                response = requests.get(f"{api_url}/bpm", timeout=5)
                if response.status_code == 200:
                    reading_data = response.json()
                    self.process_reading(reading_data)
                else:
                    self.trigger_alert(
                        AlertType.SYSTEM_ERROR,
                        AlertLevel.CRITICAL,
                        f"API request failed with status {response.status_code}"
                    )
                
            except requests.exceptions.RequestException as e:
                self.trigger_alert(
                    AlertType.SYSTEM_ERROR,
                    AlertLevel.CRITICAL,
                    f"Cannot connect to heart rate API: {str(e)}"
                )
            except Exception as e:
                self.trigger_alert(
                    AlertType.SYSTEM_ERROR,
                    AlertLevel.CRITICAL,
                    f"Monitoring error: {str(e)}"
                )
            
            time.sleep(5)  # Check every 5 seconds


# Example usage and alert callback
def print_alert_callback(alert: Alert):
    """Simple callback that prints alerts to console."""
    timestamp = alert.timestamp.strftime("%H:%M:%S")
    print(f"[{timestamp}] {alert.level.value.upper()}: {alert.message}")

def email_alert_callback(alert: Alert):
    """Callback to send email alerts (placeholder implementation)."""
    if alert.level == AlertLevel.CRITICAL:
        # Here you would implement actual email sending
        print(f"EMAIL ALERT: {alert.message}")

# Global monitor instance
health_monitor = HealthMonitor()
