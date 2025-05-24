# backend.py
import threading
import time
import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Dict, Optional
import base64

from collections import deque
from scipy.signal import butter, filtfilt, detrend, welch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow Streamlit frontend to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# — rPPG CONFIG —
BUFFER_SIZE = 200          # ~10s at 20 FPS for better stability
MIN_HZ, MAX_HZ = 0.7, 4.0  # 42–240 BPM
SAMPLING_RATE = 20         # Target FPS
CONFIDENCE_THRESHOLD = 0.6

# Global state
signal_buffer = deque(maxlen=BUFFER_SIZE)
time_buffer = deque(maxlen=BUFFER_SIZE)
latest_bpm = 0
is_camera_active = False
face_detected = False
last_update_time = 0
current_frame = None
camera_lock = threading.Lock()

# init MediaPipe FaceMesh once
mp_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=CONFIDENCE_THRESHOLD,
    min_tracking_confidence=CONFIDENCE_THRESHOLD
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
    global latest_bpm, is_camera_active, face_detected, last_update_time, current_frame
    
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
            
        # Set camera properties for consistent frame rate
        cap.set(cv2.CAP_PROP_FPS, SAMPLING_RATE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
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
                time.sleep(0.1)
                continue

            # Store current frame for streaming (thread-safe)
            with camera_lock:
                current_frame = frame.copy()

            current_time = time.time()
            frame_count += 1
            
            # Process at target sampling rate
            if current_time - last_processing_time < 1.0 / SAMPLING_RATE:
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
            if len(signal_buffer) >= BUFFER_SIZE * 0.8:  # Use 80% of buffer for more frequent updates
                try:
                    times = np.array(time_buffer)
                    if len(times) < 2:
                        continue
                        
                    # Calculate actual sampling rate
                    duration = times[-1] - times[0]
                    if duration <= 0:
                        continue
                        
                    fs = len(times) / duration
                    
                    if fs < MIN_HZ * 2:  # Nyquist criterion
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
                    sig_filtered = bandpass(sig, MIN_HZ, MAX_HZ, fs)
                    
                    # Power spectral density analysis
                    nperseg = min(len(sig_filtered) // 2, 128)
                    if nperseg < 8:
                        continue
                        
                    freqs, psd = welch(sig_filtered, fs, nperseg=nperseg, noverlap=nperseg//2)
                    
                    # Find peak in valid frequency range
                    valid_mask = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
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
                    logger.error(f"BPM computation error: {e}")
            
            # Control frame processing rate
            time.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Video loop error: {e}")
    finally:
        is_camera_active = False
        if cap:
            cap.release()
        logger.info("Camera released")

# start background thread
video_thread = threading.Thread(target=_video_loop, daemon=True)
video_thread.start()
logger.info("Video processing thread started")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "Heart rate detection server is running"}

@app.get("/status")
async def get_status():
    """Get system status"""
    global last_update_time
    current_time = time.time()
    time_since_update = current_time - last_update_time if last_update_time > 0 else float('inf')
    
    return {
        "camera_active": is_camera_active,
        "face_detected": face_detected,
        "buffer_size": len(signal_buffer),
        "max_buffer_size": BUFFER_SIZE,
        "time_since_last_update": time_since_update,
        "signal_quality": "good" if time_since_update < 2.0 else "poor"
    }

@app.get("/bpm")
async def get_bpm():
    """Return the most recent BPM estimate."""
    global last_update_time
    
    if not is_camera_active:
        raise HTTPException(status_code=503, detail="Camera not active")
    
    current_time = time.time()
    time_since_update = current_time - last_update_time if last_update_time > 0 else float('inf')
    
    # Check if data is recent enough
    if time_since_update > 5.0:  # No update for 5 seconds
        confidence = "low"
        message = "No recent face detection"
    elif time_since_update > 2.0:  # No update for 2 seconds
        confidence = "medium"
        message = "Face detection intermittent"
    else:
        confidence = "high"
        message = "Face detected"
    
    return {
        "bpm": latest_bpm,
        "confidence": confidence,
        "message": message,
        "face_detected": face_detected,
        "buffer_fill": f"{len(signal_buffer)}/{BUFFER_SIZE}",
        "timestamp": current_time
    }

def generate_frames():
    """Generate video frames from the current camera feed."""
    while is_camera_active:
        with camera_lock:
            if current_frame is not None:
                # Draw face detection indicators on frame for visual feedback
                frame_with_overlay = current_frame.copy()
                
                # Add face detection indicator
                if face_detected:
                    cv2.putText(frame_with_overlay, "Face Detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame_with_overlay, f"BPM: {latest_bpm}", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame_with_overlay, "No Face", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Add buffer status
                buffer_status = f"Buffer: {len(signal_buffer)}/{BUFFER_SIZE}"
                cv2.putText(frame_with_overlay, buffer_status, (10, 110), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame_with_overlay, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
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

@app.get("/current_frame")
async def get_current_frame():
    """Get the current frame as base64 encoded image."""
    if not is_camera_active or current_frame is None:
        raise HTTPException(status_code=503, detail="Camera not active or no frame available")
    
    with camera_lock:
        frame_copy = current_frame.copy()
        
        # Add overlay information
        if face_detected:
            cv2.putText(frame_copy, "Face Detected", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_copy, f"BPM: {latest_bpm}", (10, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame_copy, "No Face", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Encode frame as JPEG and convert to base64
        _, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "frame": frame_b64,
            "face_detected": face_detected,
            "bpm": latest_bpm,
            "timestamp": time.time()
        }
