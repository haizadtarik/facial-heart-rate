# Enhanced Facial Heart Rate Detection Backend API
import threading
import time
import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Dict, Optional, List
import base64
from datetime import datetime, timedelta
import os
from pathlib import Path

from collections import deque
from scipy.signal import butter, filtfilt, detrend, welch
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app configuration
app = FastAPI(
    title="Facial Heart Rate Detection API",
    description="A real-time heart rate detection system using facial recognition and photoplethysmography",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API responses
class HeartRateResponse(BaseModel):
    bpm: int
    confidence: str
    message: str
    face_detected: bool
    buffer_fill: str
    timestamp: float
    signal_quality: str

class SystemStatusResponse(BaseModel):
    camera_active: bool
    face_detected: bool
    buffer_size: int
    max_buffer_size: int
    time_since_last_update: float
    signal_quality: str
    api_version: str
    system_health: str

class FrameResponse(BaseModel):
    frame: str
    face_detected: bool
    bpm: int
    timestamp: float
    frame_width: int
    frame_height: int

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime_seconds: float

# System configuration
class Config:
    # rPPG CONFIG
    BUFFER_SIZE = 200          # ~10s at 20 FPS for better stability
    MIN_HZ, MAX_HZ = 0.7, 4.0  # 42–240 BPM
    SAMPLING_RATE = 20         # Target FPS
    CONFIDENCE_THRESHOLD = 0.6
    
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 20
    
    # Health monitoring
    MAX_NO_UPDATE_SECONDS = 10
    MAX_POOR_SIGNAL_SECONDS = 30

# Application startup time for uptime calculation
APP_START_TIME = datetime.now()

# Enhanced global state
signal_buffer = deque(maxlen=Config.BUFFER_SIZE)
time_buffer = deque(maxlen=Config.BUFFER_SIZE)
latest_bpm = 0
is_camera_active = False
face_detected = False
last_update_time = 0
current_frame = None
camera_lock = threading.Lock()
frame_count = 0
processing_errors = 0

# init MediaPipe FaceMesh once
mp_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=Config.CONFIDENCE_THRESHOLD,
    min_tracking_confidence=Config.CONFIDENCE_THRESHOLD
)

def bandpass(data, low, high, fs, order=4):
    """Apply bandpass filter to signal"""
    try:
        nyq = 0.5 * fs
        low_norm = low / nyq
        high_norm = high / nyq
        
        # Ensure normalized frequencies are valid
        if low_norm <= 0 or high_norm >= 1 or low_norm >= high_norm:
            logger.warning(f"Invalid filter frequencies: {low_norm}, {high_norm}")
            return data
            
        b, a = butter(order, [low_norm, high_norm], btype='band')
        return filtfilt(b, a, data)
    except Exception as e:
        logger.error(f"Bandpass filter error: {e}")
        return data

def extract_roi_signal(frame, landmarks, roi_indices):
    """Extract mean RGB signal from ROI defined by landmark indices"""
    try:
        h, w, _ = frame.shape
        points = []
        
        for idx in roi_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                points.append([int(lm.x * w), int(lm.y * h)])
        
        if len(points) < 3:
            return None
            
        points = np.array(points)
        x, y, roi_w, roi_h = cv2.boundingRect(points)
        
        # Add padding and ensure bounds
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        roi_w = min(w - x, roi_w + 2 * padding)
        roi_h = min(h - y, roi_h + 2 * padding)
        
        roi = frame[y:y+roi_h, x:x+roi_w]
        
        if roi.size == 0:
            return None
            
        # Extract green channel (most sensitive to blood volume changes)
        green_channel = roi[:, :, 1]
        return np.mean(green_channel)
        
    except Exception as e:
        logger.error(f"ROI extraction error: {e}")
        return None

def _video_loop():
    """ Continuously capture frames, extract green-channel rPPG, compute BPM. """
    global latest_bpm, is_camera_active, face_detected, last_update_time, current_frame, frame_count, processing_errors
    
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
            
        # Set camera properties for consistent frame rate
        cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        
        is_camera_active = True
        logger.info("Camera initialized successfully")
        
        # Forehead landmarks for ROI extraction
        forehead_indices = [10, 151, 9, 8, 107, 55, 8, 285, 336, 296, 334]
        
        frame_count = 0
        last_processing_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                processing_errors += 1
                time.sleep(0.1)
                continue

            # Store current frame for streaming (thread-safe)
            with camera_lock:
                current_frame = frame.copy()

            current_time = time.time()
            frame_count += 1
            
            # Process at target sampling rate
            if current_time - last_processing_time < 1.0 / Config.SAMPLING_RATE:
                continue
                
            last_processing_time = current_time

            # FaceMesh detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_mesh.process(rgb)
            
            face_detected = False
            
            if results.multi_face_landmarks:
                face_detected = True
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Extract signal from forehead region
                signal_value = extract_roi_signal(frame, landmarks, forehead_indices)
                
                if signal_value is not None:
                    signal_buffer.append(signal_value)
                    time_buffer.append(current_time)
                    last_update_time = current_time

            # Compute BPM when we have sufficient data
            if len(signal_buffer) >= Config.BUFFER_SIZE * 0.8:  # Use 80% of buffer for more frequent updates
                try:
                    times = np.array(time_buffer)
                    if len(times) < 2:
                        continue
                        
                    # Calculate actual sampling rate
                    duration = times[-1] - times[0]
                    if duration <= 0:
                        continue
                        
                    fs = len(times) / duration
                    
                    if fs < Config.MIN_HZ * 2:  # Nyquist criterion
                        logger.warning(f"Sampling rate too low: {fs:.2f} Hz")
                        continue
                    
                    # Process signal
                    sig = np.array(signal_buffer)
                    if len(sig) < 10:
                        continue
                    
                    # Normalize and detrend
                    if sig.std() > 0:
                        sig = (sig - sig.mean()) / sig.std()
                    sig = detrend(sig)
                    
                    # Apply bandpass filter
                    sig_filtered = bandpass(sig, Config.MIN_HZ, Config.MAX_HZ, fs)
                    
                    # Power spectral density analysis
                    nperseg = min(len(sig_filtered) // 2, 128)
                    if nperseg < 8:
                        continue
                        
                    freqs, psd = welch(sig_filtered, fs, nperseg=nperseg, noverlap=nperseg//2)
                    
                    # Find peak in valid frequency range
                    valid_mask = (freqs >= Config.MIN_HZ) & (freqs <= Config.MAX_HZ)
                    if not valid_mask.any():
                        continue
                        
                    valid_freqs = freqs[valid_mask]
                    valid_psd = psd[valid_mask]
                    
                    if len(valid_psd) > 0:
                        peak_idx = np.argmax(valid_psd)
                        peak_freq = valid_freqs[peak_idx]
                        new_bpm = int(peak_freq * 60)
                        
                        # Smooth BPM updates to reduce noise
                        if latest_bpm == 0:
                            latest_bpm = new_bpm
                        else:
                            # Apply simple moving average
                            alpha = 0.3
                            latest_bpm = int(alpha * new_bpm + (1 - alpha) * latest_bpm)
                        
                        logger.info(f"BPM updated: {latest_bpm}, Peak freq: {peak_freq:.2f} Hz")
                        
                except Exception as e:
                    processing_errors += 1
                    logger.error(f"BPM computation error: {e}")
            
            # Control frame processing rate
            time.sleep(0.01)
            
    except Exception as e:
        processing_errors += 1
        logger.error(f"Video loop error: {e}")
    finally:
        is_camera_active = False
        if cap:
            cap.release()
        logger.info("Camera released")

# Application startup and lifecycle management

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Facial Heart Rate Detection API v2.0.0")
    logger.info(f"Camera resolution: {Config.CAMERA_WIDTH}x{Config.CAMERA_HEIGHT}")
    logger.info(f"Target FPS: {Config.CAMERA_FPS}")
    logger.info(f"BPM range: {Config.MIN_HZ*60:.0f}-{Config.MAX_HZ*60:.0f}")
    
    # Start background video processing thread
    video_thread = threading.Thread(target=_video_loop, daemon=True)
    video_thread.start()
    logger.info("Video processing thread started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    global is_camera_active
    logger.info("Shutting down application...")
    is_camera_active = False
    logger.info("Application shutdown complete")

# For development and testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production
        workers=1,     # Single worker for camera access
        log_level="info"
    )

# Enhanced API endpoints with Pydantic models

@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Health check endpoint with system information"""
    uptime = (datetime.now() - APP_START_TIME).total_seconds()
    return HealthCheckResponse(
        status="Heart rate detection server is running",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        uptime_seconds=uptime
    )

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Detailed health check endpoint"""
    uptime = (datetime.now() - APP_START_TIME).total_seconds()
    return HealthCheckResponse(
        status="healthy" if is_camera_active else "degraded",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        uptime_seconds=uptime
    )

@app.get("/status", response_model=SystemStatusResponse)
async def get_status():
    """Get comprehensive system status"""
    global last_update_time, processing_errors
    current_time = time.time()
    time_since_update = current_time - last_update_time if last_update_time > 0 else float('inf')
    
    # Determine signal quality
    if time_since_update > Config.MAX_POOR_SIGNAL_SECONDS:
        signal_quality = "poor"
        system_health = "degraded"
    elif time_since_update > 5.0:
        signal_quality = "fair"
        system_health = "fair"
    else:
        signal_quality = "good"
        system_health = "healthy"
    
    return SystemStatusResponse(
        camera_active=is_camera_active,
        face_detected=face_detected,
        buffer_size=len(signal_buffer),
        max_buffer_size=Config.BUFFER_SIZE,
        time_since_last_update=time_since_update,
        signal_quality=signal_quality,
        api_version="2.0.0",
        system_health=system_health
    )

@app.get("/bpm", response_model=HeartRateResponse)
async def get_bpm():
    """Return the most recent BPM estimate with enhanced metadata."""
    global last_update_time
    
    if not is_camera_active:
        raise HTTPException(status_code=503, detail="Camera not active")
    
    current_time = time.time()
    time_since_update = current_time - last_update_time if last_update_time > 0 else float('inf')
    
    # Determine confidence and signal quality
    if time_since_update > Config.MAX_NO_UPDATE_SECONDS:
        confidence = "low"
        signal_quality = "poor"
        message = "No recent face detection"
    elif time_since_update > 5.0:
        confidence = "medium"
        signal_quality = "fair"
        message = "Face detection intermittent"
    else:
        confidence = "high"
        signal_quality = "good"
        message = "Face detected"
    
    return HeartRateResponse(
        bpm=latest_bpm,
        confidence=confidence,
        message=message,
        face_detected=face_detected,
        buffer_fill=f"{len(signal_buffer)}/{Config.BUFFER_SIZE}",
        timestamp=current_time,
        signal_quality=signal_quality
    )

@app.get("/metrics")
async def get_metrics():
    """Get system metrics for monitoring"""
    global processing_errors, frame_count
    uptime = (datetime.now() - APP_START_TIME).total_seconds()
    
    return {
        "uptime_seconds": uptime,
        "frames_processed": frame_count,
        "processing_errors": processing_errors,
        "error_rate": processing_errors / max(frame_count, 1),
        "buffer_utilization": len(signal_buffer) / Config.BUFFER_SIZE,
        "camera_active": is_camera_active,
        "face_detected": face_detected,
        "current_bpm": latest_bpm
    }

def generate_frames():
    """Generate video frames from the current camera feed."""
    while is_camera_active:
        with camera_lock:
            if current_frame is not None:
                # Draw enhanced overlay information on frame
                frame_with_overlay = current_frame.copy()
                h, w = frame_with_overlay.shape[:2]
                
                # Add face detection indicator
                if face_detected:
                    cv2.putText(frame_with_overlay, "Face Detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame_with_overlay, f"BPM: {latest_bpm}", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Add green border for face detected
                    cv2.rectangle(frame_with_overlay, (5, 5), (w-5, h-5), (0, 255, 0), 3)
                else:
                    cv2.putText(frame_with_overlay, "No Face Detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Add red border for no face
                    cv2.rectangle(frame_with_overlay, (5, 5), (w-5, h-5), (0, 0, 255), 3)
                
                # Add buffer status and timestamp
                buffer_status = f"Buffer: {len(signal_buffer)}/{Config.BUFFER_SIZE}"
                cv2.putText(frame_with_overlay, buffer_status, (10, 110), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame_with_overlay, timestamp, (10, h-20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame_with_overlay, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.05)  # ~20 FPS

@app.get("/video_feed")
async def video_feed():
    """Video streaming endpoint with MJPEG format."""
    if not is_camera_active:
        raise HTTPException(status_code=503, detail="Camera not active")
    
    return StreamingResponse(generate_frames(), 
                           media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/current_frame", response_model=FrameResponse)
async def get_current_frame():
    """Get the current frame as base64 encoded image with metadata."""
    if not is_camera_active or current_frame is None:
        raise HTTPException(status_code=503, detail="Camera not active or no frame available")
    
    with camera_lock:
        frame_copy = current_frame.copy()
        h, w = frame_copy.shape[:2]
        
        # Add overlay information
        if face_detected:
            cv2.putText(frame_copy, "Face Detected", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_copy, f"BPM: {latest_bpm}", (10, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame_copy, (5, 5), (w-5, h-5), (0, 255, 0), 3)
        else:
            cv2.putText(frame_copy, "No Face Detected", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame_copy, (5, 5), (w-5, h-5), (0, 0, 255), 3)
        
        # Add buffer and timestamp info
        buffer_status = f"Buffer: {len(signal_buffer)}/{Config.BUFFER_SIZE}"
        cv2.putText(frame_copy, buffer_status, (10, 110), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame_copy, timestamp, (10, h-20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode frame as JPEG and convert to base64
        _, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return FrameResponse(
            frame=frame_b64,
            face_detected=face_detected,
            bpm=latest_bpm,
            timestamp=time.time(),
            frame_width=w,
            frame_height=h
        )
