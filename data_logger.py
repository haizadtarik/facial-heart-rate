"""
Data logging module for heart rate detection system.
Stores readings in SQLite database for historical analysis.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

class HeartRateLogger:
    def __init__(self, db_path: str = "heart_rate_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS heart_rate_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    bpm INTEGER,
                    confidence TEXT,
                    face_detected BOOLEAN,
                    buffer_fill TEXT,
                    signal_quality TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    avg_bpm REAL,
                    min_bpm INTEGER,
                    max_bpm INTEGER,
                    total_readings INTEGER
                )
            """)
            conn.commit()
    
    def log_reading(self, bpm: int, confidence: str, face_detected: bool, 
                   buffer_fill: str, signal_quality: str = "unknown"):
        """Log a single heart rate reading."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO heart_rate_readings 
                (bpm, confidence, face_detected, buffer_fill, signal_quality)
                VALUES (?, ?, ?, ?, ?)
            """, (bpm, confidence, face_detected, buffer_fill, signal_quality))
            conn.commit()
    
    def get_recent_readings(self, limit: int = 100) -> List[Dict]:
        """Get recent heart rate readings."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, bpm, confidence, face_detected, signal_quality
                FROM heart_rate_readings 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_session_stats(self, hours: int = 24) -> Dict:
        """Get statistics for the last N hours."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_readings,
                    AVG(bpm) as avg_bpm,
                    MIN(bpm) as min_bpm,
                    MAX(bpm) as max_bpm,
                    COUNT(CASE WHEN confidence = 'high' THEN 1 END) as high_confidence_readings
                FROM heart_rate_readings 
                WHERE timestamp > datetime('now', '-{} hours')
                AND face_detected = 1
            """.format(hours))
            
            result = cursor.fetchone()
            if result:
                return {
                    'total_readings': result[0] or 0,
                    'avg_bpm': round(result[1] or 0, 1),
                    'min_bpm': result[2] or 0,
                    'max_bpm': result[3] or 0,
                    'high_confidence_readings': result[4] or 0,
                    'hours': hours
                }
            return {}
    
    def export_data(self, filename: str, hours: int = 24):
        """Export recent data to JSON file."""
        readings = self.get_recent_readings(1000)
        stats = self.get_session_stats(hours)
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'statistics': stats,
            'readings': readings
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)