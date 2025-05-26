# Enhanced Facial Heart Rate Detection Backend API - Image Based Processing
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
import uuid
import asyncio

from collections import deque
from scipy.signal import butter, filtfilt, detrend, welch
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app configuration
app = FastAPI(
    title="Facial Heart Rate Detection API - Image Based",
    description="A heart rate detection system using facial recognition and photoplethysmography from uploaded images",
    version="2.1.0",
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
    bpm: Optional[int]
    confidence: str
    message: str
    face_detected: bool
    buffer_fill: str
    timestamp: float
    signal_quality: str

class ImageAnalysisRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    session_id: Optional[str] = Field(None, description="Session ID for tracking analysis sessions")

class ImageAnalysisResponse(BaseModel):
    bpm: Optional[int]
    confidence: str
    message: str
    face_detected: bool
    timestamp: float
    signal_quality: str
    session_id: str
    processed_frame: Optional[str] = Field(None, description="Base64 encoded processed frame with overlays")

class SessionStatusResponse(BaseModel):
    session_id: str
    is_active: bool
    buffer_size: int
    max_buffer_size: int
    total_frames_processed: int
    time_since_last_frame: float
    current_bpm: Optional[int]
    signal_quality: str

class BatchAnalysisRequest(BaseModel):
    images: List[str] = Field(..., description="List of base64 encoded images")
    session_id: Optional[str] = Field(None, description="Session ID for tracking analysis sessions")
    frame_rate: Optional[float] = Field(20.0, description="Frame rate for temporal analysis")

class BatchAnalysisResponse(BaseModel):
    session_id: str
    total_frames: int
    faces_detected: int
    final_bpm: Optional[int]
    confidence: str
    signal_quality: str
    timestamp: float

class SystemStatusResponse(BaseModel):
    active_sessions: int
    total_sessions_created: int
    total_frames_processed: int
    processing_errors: int
    error_rate: float
    api_version: str
    system_health: str

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
    
    # Session management
    SESSION_TIMEOUT = 300      # 5 minutes
    MAX_SESSIONS = 100         # Maximum concurrent sessions
    
    # Health monitoring
    MAX_NO_UPDATE_SECONDS = 10
    MAX_POOR_SIGNAL_SECONDS = 30

# Application startup time for uptime calculation
APP_START_TIME = datetime.now()

# Enhanced global state for image-based processing
signal_buffers = {}      # Dictionary to store signal buffers per session
time_buffers = {}        # Dictionary to store time buffers per session
session_data = {}        # Dictionary to store session metadata
active_sessions = set()  # Set of active session IDs
processing_errors = 0
total_frames_processed = 0
total_sessions_created = 0

# Thread lock for session management
session_lock = threading.Lock()

# Initialize MediaPipe FaceMesh once
mp_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,  # Changed to True for image processing
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

def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image string to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Could not decode image")
            
        return frame
        
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

def encode_frame_to_base64(frame: np.ndarray) -> str:
    """Encode OpenCV frame to base64 string"""
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        return frame_b64
    except Exception as e:
        logger.error(f"Frame encoding error: {e}")
        return ""

def create_session() -> str:
    """Create a new analysis session"""
    with session_lock:
        session_id = str(uuid.uuid4())
        signal_buffers[session_id] = deque(maxlen=Config.BUFFER_SIZE)
        time_buffers[session_id] = deque(maxlen=Config.BUFFER_SIZE)
        session_data[session_id] = {
            'created_at': datetime.now(),
            'last_update': datetime.now(),
            'frames_processed': 0,
            'faces_detected': 0,
            'latest_bpm': None
        }
        active_sessions.add(session_id)
        global total_sessions_created
        total_sessions_created += 1
        logger.info(f"Created new session: {session_id}")
        return session_id

def cleanup_expired_sessions():
    """Remove expired sessions"""
    current_time = datetime.now()
    with session_lock:
        expired_sessions = []
        for session_id in active_sessions:
            if session_id in session_data:
                last_update = session_data[session_id]['last_update']
                if (current_time - last_update).total_seconds() > Config.SESSION_TIMEOUT:
                    expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            logger.info(f"Cleaning up expired session: {session_id}")
            active_sessions.discard(session_id)
            signal_buffers.pop(session_id, None)
            time_buffers.pop(session_id, None)
            session_data.pop(session_id, None)

def compute_bpm_for_session(session_id: str) -> Optional[int]:
    """Compute BPM for a specific session"""
    try:
        if session_id not in signal_buffers or session_id not in time_buffers:
            return None
            
        signal_buffer = signal_buffers[session_id]
        time_buffer = time_buffers[session_id]
        
        if len(signal_buffer) < Config.BUFFER_SIZE * 0.6:  # Need at least 60% of buffer
            return None
            
        times = np.array(time_buffer)
        if len(times) < 2:
            return None
            
        # Calculate actual sampling rate
        duration = times[-1] - times[0]
        if duration <= 0:
            return None
            
        fs = len(times) / duration
        
        if fs < Config.MIN_HZ * 2:  # Nyquist criterion
            logger.warning(f"Sampling rate too low: {fs:.2f} Hz for session {session_id}")
            return None
        
        # Process signal
        sig = np.array(signal_buffer)
        if len(sig) < 10:
            return None
        
        # Normalize and detrend
        if sig.std() > 0:
            sig = (sig - sig.mean()) / sig.std()
        sig = detrend(sig)
        
        # Apply bandpass filter
        sig_filtered = bandpass(sig, Config.MIN_HZ, Config.MAX_HZ, fs)
        
        # Power spectral density analysis
        nperseg = min(len(sig_filtered) // 2, 128)
        if nperseg < 8:
            return None
            
        freqs, psd = welch(sig_filtered, fs, nperseg=nperseg, noverlap=nperseg//2)
        
        # Find peak in valid frequency range
        valid_mask = (freqs >= Config.MIN_HZ) & (freqs <= Config.MAX_HZ)
        if not valid_mask.any():
            return None
            
        valid_freqs = freqs[valid_mask]
        valid_psd = psd[valid_mask]
        
        if len(valid_psd) > 0:
            peak_idx = np.argmax(valid_psd)
            peak_freq = valid_freqs[peak_idx]
            bpm = int(peak_freq * 60)
            
            logger.info(f"BPM computed for session {session_id}: {bpm}, Peak freq: {peak_freq:.2f} Hz")
            return bpm
            
    except Exception as e:
        global processing_errors
        processing_errors += 1
        logger.error(f"BPM computation error for session {session_id}: {e}")
        return None

# Background task to cleanup expired sessions
async def cleanup_task():
    """Background task to periodically cleanup expired sessions"""
    while True:
        try:
            cleanup_expired_sessions()
            await asyncio.sleep(60)  # Run every minute
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
            await asyncio.sleep(60)

# Application startup and lifecycle management
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Facial Heart Rate Detection API v2.1.0 (Image Based)")
    logger.info(f"BPM range: {Config.MIN_HZ*60:.0f}-{Config.MAX_HZ*60:.0f}")
    logger.info(f"Buffer size: {Config.BUFFER_SIZE}")
    logger.info(f"Session timeout: {Config.SESSION_TIMEOUT}s")
    
    # Start background cleanup task
    asyncio.create_task(cleanup_task())
    logger.info("Background cleanup task started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down application...")
    with session_lock:
        active_sessions.clear()
        signal_buffers.clear()
        time_buffers.clear()
        session_data.clear()
    logger.info("Application shutdown complete")

# API endpoints

@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Health check endpoint with system information"""
    uptime = (datetime.now() - APP_START_TIME).total_seconds()
    return HealthCheckResponse(
        status="Heart rate detection server is running (Image Based)",
        timestamp=datetime.now().isoformat(),
        version="2.1.0",
        uptime_seconds=uptime
    )

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Detailed health check endpoint"""
    uptime = (datetime.now() - APP_START_TIME).total_seconds()
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.1.0",
        uptime_seconds=uptime
    )

@app.get("/status", response_model=SystemStatusResponse)
async def get_status():
    """Get comprehensive system status"""
    cleanup_expired_sessions()
    error_rate = processing_errors / max(total_frames_processed, 1)
    
    return SystemStatusResponse(
        active_sessions=len(active_sessions),
        total_sessions_created=total_sessions_created,
        total_frames_processed=total_frames_processed,
        processing_errors=processing_errors,
        error_rate=error_rate,
        api_version="2.1.0",
        system_health="healthy" if error_rate < 0.1 else "degraded"
    )

@app.post("/create-session")
async def create_analysis_session():
    """Create a new analysis session"""
    if len(active_sessions) >= Config.MAX_SESSIONS:
        cleanup_expired_sessions()
        if len(active_sessions) >= Config.MAX_SESSIONS:
            raise HTTPException(status_code=429, detail="Maximum number of sessions reached")
    
    session_id = create_session()
    return {"session_id": session_id, "message": "Session created successfully"}

@app.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """Get status of a specific session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_info = session_data[session_id]
    current_time = datetime.now()
    time_since_last = (current_time - session_info['last_update']).total_seconds()
    
    # Determine signal quality
    if time_since_last > Config.MAX_POOR_SIGNAL_SECONDS:
        signal_quality = "poor"
    elif time_since_last > 5.0:
        signal_quality = "fair"
    else:
        signal_quality = "good"
    
    return SessionStatusResponse(
        session_id=session_id,
        is_active=True,
        buffer_size=len(signal_buffers.get(session_id, [])),
        max_buffer_size=Config.BUFFER_SIZE,
        total_frames_processed=session_info['frames_processed'],
        time_since_last_frame=time_since_last,
        current_bpm=session_info['latest_bpm'],
        signal_quality=signal_quality
    )

@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(request: ImageAnalysisRequest):
    """Analyze a single image for heart rate detection"""
    global total_frames_processed
    
    # Create session if not provided
    session_id = request.session_id or create_session()
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    try:
        # Decode the image
        frame = decode_base64_image(request.image_data)
        h, w = frame.shape[:2]
        
        current_time = time.time()
        total_frames_processed += 1
        
        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_mesh.process(rgb)
        
        face_detected = False
        processed_frame = frame.copy()
        
        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Forehead landmarks for ROI extraction
            forehead_indices = [10, 151, 9, 8, 107, 55, 8, 285, 336, 296, 334]
            
            # Extract signal from forehead region
            signal_value = extract_roi_signal(frame, landmarks, forehead_indices)
            
            if signal_value is not None:
                signal_buffers[session_id].append(signal_value)
                time_buffers[session_id].append(current_time)
                
                # Update session data
                session_data[session_id]['last_update'] = datetime.now()
                session_data[session_id]['frames_processed'] += 1
                session_data[session_id]['faces_detected'] += 1
        
        # Compute BPM
        bpm = compute_bpm_for_session(session_id)
        if bpm is not None:
            session_data[session_id]['latest_bpm'] = bpm
        
        # Create processed frame with overlays
        if face_detected:
            cv2.putText(processed_frame, "Face Detected", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if bpm:
                cv2.putText(processed_frame, f"BPM: {bpm}", (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(processed_frame, (5, 5), (w-5, h-5), (0, 255, 0), 3)
        else:
            cv2.putText(processed_frame, "No Face Detected", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(processed_frame, (5, 5), (w-5, h-5), (0, 0, 255), 3)
        
        # Add buffer status
        buffer_size = len(signal_buffers[session_id])
        buffer_status = f"Buffer: {buffer_size}/{Config.BUFFER_SIZE}"
        cv2.putText(processed_frame, buffer_status, (10, 110), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Determine confidence and signal quality
        session_info = session_data[session_id]
        time_since_last = (datetime.now() - session_info['last_update']).total_seconds()
        
        if not face_detected:
            confidence = "low"
            signal_quality = "poor"
            message = "No face detected in image"
        elif buffer_size < Config.BUFFER_SIZE * 0.6:
            confidence = "low"
            signal_quality = "poor"
            message = "Insufficient data for reliable measurement"
        elif bpm is None:
            confidence = "medium"
            signal_quality = "fair"
            message = "Face detected, building signal buffer"
        else:
            confidence = "high"
            signal_quality = "good"
            message = "Heart rate detected successfully"
        
        return ImageAnalysisResponse(
            bpm=bpm,
            confidence=confidence,
            message=message,
            face_detected=face_detected,
            timestamp=current_time,
            signal_quality=signal_quality,
            session_id=session_id,
            processed_frame=encode_frame_to_base64(processed_frame)
        )
        
    except Exception as e:
        global processing_errors
        processing_errors += 1
        logger.error(f"Image analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {e}")

@app.post("/analyze-batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """Analyze a batch of images for heart rate detection"""
    global total_frames_processed
    
    if len(request.images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")
    
    if len(request.images) > 100:
        raise HTTPException(status_code=400, detail="Too many images (max 100)")
    
    # Create session if not provided
    session_id = request.session_id or create_session()
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    try:
        faces_detected = 0
        frame_interval = 1.0 / request.frame_rate if request.frame_rate > 0 else 0.05
        start_time = time.time()
        
        for i, image_data in enumerate(request.images):
            try:
                # Decode the image
                frame = decode_base64_image(image_data)
                
                current_time = start_time + (i * frame_interval)
                total_frames_processed += 1
                
                # Process with MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_mesh.process(rgb)
                
                if results.multi_face_landmarks:
                    faces_detected += 1
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Forehead landmarks for ROI extraction
                    forehead_indices = [10, 151, 9, 8, 107, 55, 8, 285, 336, 296, 334]
                    
                    # Extract signal from forehead region
                    signal_value = extract_roi_signal(frame, landmarks, forehead_indices)
                    
                    if signal_value is not None:
                        signal_buffers[session_id].append(signal_value)
                        time_buffers[session_id].append(current_time)
                        
                        # Update session data
                        session_data[session_id]['last_update'] = datetime.now()
                        session_data[session_id]['frames_processed'] += 1
                        session_data[session_id]['faces_detected'] += 1
                        
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                continue
        
        # Compute final BPM
        final_bpm = compute_bpm_for_session(session_id)
        if final_bpm is not None:
            session_data[session_id]['latest_bpm'] = final_bpm
        
        # Determine confidence and signal quality
        buffer_size = len(signal_buffers[session_id])
        face_detection_rate = faces_detected / len(request.images)
        
        if face_detection_rate < 0.3:
            confidence = "low"
            signal_quality = "poor"
        elif face_detection_rate < 0.7 or buffer_size < Config.BUFFER_SIZE * 0.6:
            confidence = "medium"
            signal_quality = "fair"
        else:
            confidence = "high"
            signal_quality = "good"
        
        return BatchAnalysisResponse(
            session_id=session_id,
            total_frames=len(request.images),
            faces_detected=faces_detected,
            final_bpm=final_bpm,
            confidence=confidence,
            signal_quality=signal_quality,
            timestamp=time.time()
        )
        
    except Exception as e:
        global processing_errors
        processing_errors += 1
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {e}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete an analysis session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    with session_lock:
        active_sessions.discard(session_id)
        signal_buffers.pop(session_id, None)
        time_buffers.pop(session_id, None)
        session_data.pop(session_id, None)
    
    return {"message": f"Session {session_id} deleted successfully"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    cleanup_expired_sessions()
    sessions = []
    
    for session_id in active_sessions:
        if session_id in session_data:
            session_info = session_data[session_id]
            sessions.append({
                "session_id": session_id,
                "created_at": session_info['created_at'].isoformat(),
                "last_update": session_info['last_update'].isoformat(),
                "frames_processed": session_info['frames_processed'],
                "faces_detected": session_info['faces_detected'],
                "current_bpm": session_info['latest_bpm']
            })
    
    return {"active_sessions": sessions, "total_count": len(sessions)}

@app.get("/metrics")
async def get_metrics():
    """Get system metrics for monitoring"""
    cleanup_expired_sessions()
    uptime = (datetime.now() - APP_START_TIME).total_seconds()
    error_rate = processing_errors / max(total_frames_processed, 1)
    
    return {
        "uptime_seconds": uptime,
        "total_frames_processed": total_frames_processed,
        "processing_errors": processing_errors,
        "error_rate": error_rate,
        "active_sessions": len(active_sessions),
        "total_sessions_created": total_sessions_created,
        "system_health": "healthy" if error_rate < 0.1 else "degraded"
    }

# For development and testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server_image_based:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
