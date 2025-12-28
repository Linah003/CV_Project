"""
Draa - Smart Violence Detection System
Complete Application with Swapped Color Scheme
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import json
import base64
from datetime import datetime
from collections import deque
import shutil
import uvicorn
import time


# =============== CONFIGURATION ===============
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
SEQUENCE_LENGTH = 15
CLASSES_LIST = ["Normal", "Violence"] 
TEMPORAL_WINDOW_SIZE = 5
TEMPORAL_THRESHOLD = 0.6
MODEL_PATH = "./model.keras"
# =============== INITIALIZE APP ===============
app = FastAPI(title="Draa - Violence Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("incidents", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# =============== LOAD MODEL ===============
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# =============== DATA STORAGE ===============
INCIDENTS_FILE = "incidents/incidents.json"

def load_incidents():
    if os.path.exists(INCIDENTS_FILE):
        with open(INCIDENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_incident(incident):
    incidents = load_incidents()
    incidents.insert(0, incident)
    with open(INCIDENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(incidents, f, indent=2, ensure_ascii=False)

# =============== TEMPORAL CONSISTENCY ===============
class TemporalConsistencyChecker:
    def __init__(self, window_size=TEMPORAL_WINDOW_SIZE, threshold=TEMPORAL_THRESHOLD):
        self.window_size = window_size
        self.threshold = threshold
        self.history = deque(maxlen=window_size)
    
    def update(self, prediction_probs):
        self.history.append(prediction_probs)
        if len(self.history) < self.window_size:
            return prediction_probs
        avg_probs = np.mean(list(self.history), axis=0)
        if np.max(avg_probs) < self.threshold:
            return prediction_probs
        return avg_probs
    
    def reset(self):
        self.history.clear()

class VideoRecorder:
    def __init__(self, pre_buffer_seconds=5, post_buffer_seconds=5, frame_rate=10):
        self.pre_buffer_seconds = pre_buffer_seconds
        self.post_buffer_seconds = post_buffer_seconds
        self.frame_rate = frame_rate
        self.max_buffer_frames = pre_buffer_seconds * frame_rate
        
        # Circular buffer for pre-fight frames
        self.frame_buffer = deque(maxlen=self.max_buffer_frames)
        
        # Recording state
        self.is_recording = False
        self.recording_started = False
        self.fight_ended_time = None
        self.video_writer = None
        self.output_path = None
        self.frame_count = 0
        self.current_timestamp = None

    
    def add_frame(self, frame):
        """Add frame to circular buffer"""
        self.frame_buffer.append(frame.copy())
    
    def start_recording(self, timestamp):
        """Start recording when fight is detected"""
        if self.recording_started:
            return None
        
        self.recording_started = True
        self.is_recording = True
        self.fight_ended_time = None
        self.current_timestamp = timestamp

        # Create output path
        filename = f"camera_{timestamp}.mp4"
        self.output_path = f"uploads/{filename}"
        
        # Get frame dimensions from buffer
        if len(self.frame_buffer) > 0:
            height, width = self.frame_buffer[0].shape[:2]
        else:
            height, width = 480, 640
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.frame_rate, (width, height)
        )
        
        # Write buffered frames (pre-fight)
        print(f"📹 Writing {len(self.frame_buffer)} pre-fight frames...")
        for buffered_frame in self.frame_buffer:
            self.video_writer.write(buffered_frame)
            self.frame_count += 1
        
        print(f"🎬 Started recording: {self.output_path}")
        return self.output_path
    
    def add_recording_frame(self, frame):
        """Add frame to active recording"""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)
            self.frame_count += 1
    
    def mark_fight_ended(self):
        """Mark that fight has ended, continue recording for post-buffer"""
        if self.is_recording and self.fight_ended_time is None:
            self.fight_ended_time = time.time()
            print(f"⏱️ Fight ended, recording for {self.post_buffer_seconds} more seconds...")
    
    def should_stop_recording(self):
        """Check if we should stop recording"""
        if not self.is_recording:
            return False
        if self.fight_ended_time is not None:
            elapsed = time.time() - self.fight_ended_time
            if elapsed >= self.post_buffer_seconds:
                return True
        return False
    
    def stop_recording(self):
        """Stop recording and finalize video"""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"✅ Recording stopped: {self.frame_count} frames saved to {self.output_path}")
        
        output = self.output_path
        timestamp = self.current_timestamp
        self.is_recording = False
        self.recording_started = False
        self.fight_ended_time = None
        self.video_writer = None
        self.output_path = None
        self.frame_count = 0
        self.current_timestamp = None
        return output, timestamp

# =============== FRAME EXTRACTION ===============
def frames_extraction(video_path):
    """Extract frames matching the training preprocessing"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = max(1, (total_frames // SEQUENCE_LENGTH) - 1)

        for _ in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Match training preprocessing exactly
            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frame = frame / 255.0  # Normalize to [0, 1]
            frames.append(frame)
            
            # Skip frames
            for _ in range(skip_frames):
                cap.grab()

        cap.release()

        # Pad with last frame if needed
        if len(frames) > 0:
            last_frame = frames[-1]
            while len(frames) < SEQUENCE_LENGTH:
                frames.append(last_frame.copy())
        else:
            frames = [np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3)) 
                     for _ in range(SEQUENCE_LENGTH)]

        return np.array(frames)
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return np.zeros((SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
# =============== PREDICTION ===============

def predict_video(video_path):
    """Predict using the trained model"""
    if model is None:
        return {"error": "Model not loaded"}
    
    frames = frames_extraction(video_path)
    frames = np.expand_dims(frames, axis=0)  # Shape: (1, 15, 128, 128, 3)
    
    # Get prediction - output is probability of Violence (class 1)
    prediction = model.predict(frames, verbose=0)[0][0]  # Single sigmoid output
    
    # Threshold at 0.5
    is_violence = prediction >= 0.5
    class_name = 'Violence' if is_violence else 'Normal'
    
    return {
        'class': class_name,
        'confidence': float(prediction if is_violence else 1 - prediction),
        'probabilities': {
            'Normal': float(1 - prediction),
            'Violence': float(prediction)
        }
    }

# =============== HTML TEMPLATES ===============

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Draa - Smart Violence Detection System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: white;
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        .logo {
            text-align: center;
            margin-bottom: 50px;
            animation: fadeIn 1s;
        }
        .logo-image {
            max-width: 300px;
            height: auto;
        }
        .logo-subtitle {
            color: rgba(14, 107, 101, 0.98);
            font-size: 22px;
            font-weight: 600;
            text-shadow: 2px 2px 3px rgba(0,0,0,0.2);
            letter-spacing: 0.5px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .nav-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 35px;
        }
        .nav-button {
            padding: 14px 35px;
            background: #0E6B65;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            color: #ffffff;
            transition: all 0.3s ease;
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        .nav-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.25);
            background: #0E6B65;
        }
        .card {
            background: #0E6B65;
            border-radius: 25px;
            padding: 40px;
            margin-bottom: 20px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.2);
            animation: slideUp 0.6s ease;
        }
        .card h2 {
            color: #ffffff;
            margin-bottom: 30px;
            font-size: 28px;
            text-align: center;
            font-weight: 700;
        }
        .options {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin-bottom: 30px;
        }
        .option-card {
            background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
            color: #0E6B65;
            padding: 50px 40px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.4s ease;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .option-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(14, 107, 101, 0.2), transparent);
            transition: left 0.5s;
        }
        .option-card:hover::before {
            left: 100%;
        }
        .option-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 15px 35px rgba(0,0,0,0.35);
        }
        .option-card h3 { 
            font-size: 24px; 
            margin-bottom: 12px;
            font-weight: 700;
        }
        .option-card p { 
            opacity: 0.85; 
            font-size: 16px;
            line-height: 1.5;
        }
        .upload-area {
            border: 3px dashed #ffffff;
            border-radius: 20px;
            padding: 60px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            display: none;
            background: rgba(255, 255, 255, 0.03);
        }
        .upload-area.active { display: block; }
        .upload-area:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: #f5f5f5;
            transform: scale(1.01);
        }
        .upload-area h3 {
            color: #ffffff;
            font-size: 20px;
            margin-bottom: 20px;
        }
        .camera-view {
            display: none;
            text-align: center;
        }
        .camera-view.active { display: block; }
        #video {
            width: 100%;
            max-width: 800px;
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
            background: #000;
        }
        .status-bar {
            margin-top: 25px;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-weight: 700;
            font-size: 18px;
            display: none;
            animation: slideDown 0.4s ease;
        }
        .status-bar.safe {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            border: 2px solid #b1dfbb;
            box-shadow: 0 5px 15px rgba(21, 87, 36, 0.2);
        }
        .status-bar.danger {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            border: 2px solid #f1b0b7;
            animation: pulse 1.5s infinite, slideDown 0.4s ease;
            box-shadow: 0 5px 20px rgba(114, 28, 36, 0.3);
        }
        .status-bar.active { display: block; }
        .confidence {
            margin-top: 12px;
            font-size: 15px;
            opacity: 0.85;
            font-weight: 600;
        }
        .controls {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 25px;
        }
        .btn {
            padding: 14px 35px;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-size: 17px;
            font-weight: 700;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
            color: #0E6B65;
            box-shadow: 0 5px 15px rgba(255, 255, 255, 0.4);
        }
        .btn-primary:hover {
            box-shadow: 0 8px 25px rgba(255, 255, 255, 0.5);
        }
        .btn-danger { 
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
            color: #ffffff;
            box-shadow: 0 5px 15px rgba(220, 53, 69, 0.4);
        }
        .btn-danger:hover {
            box-shadow: 0 8px 25px rgba(220, 53, 69, 0.5);
        }
        .btn:hover {
            transform: translateY(-3px);
        }
        .btn:disabled { 
            opacity: 0.5; 
            cursor: not-allowed; 
            transform: none;
            box-shadow: none;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
        }
        .loading.active { display: block; }
        .spinner {
            border: 5px solid rgba(255, 255, 255, 0.2);
            border-top: 5px solid #ffffff;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        .loading p {
            margin-top: 20px;
            font-size: 16px;
            color: #ffffff;
            font-weight: 600;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.9; transform: scale(0.98); }
        }
        @media (max-width: 768px) {
            .options { grid-template-columns: 1fr; }
            .logo-image { max-width: 300px; }
            .card { padding: 25px; }
            .option-card { padding: 35px 25px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">
            <img src="/static/logo.webp" alt="Draa Logo" class="logo-image" onerror="this.style.display='none'">
            <div class="logo-subtitle">Smart Violence Detection System</div>
        </div>

        <div class="nav-buttons">
            <button class="nav-button" onclick="window.location.href='/incidents'">
                📊 Incidents Log
            </button>
        </div>

        <div class="card">
            <h2>Choose Detection Method</h2>
            
            <div class="options">
                <div class="option-card" onclick="activateUpload()">
                    <h3>📤 Upload Video</h3>
                    <p>Upload a video from your device for analysis</p>
                </div>
                <div class="option-card" onclick="activateCamera()">
                    <h3>📹 Live Camera</h3>
                    <p>Real-time detection via your camera</p>
                </div>
            </div>

            <div id="uploadArea" class="upload-area">
                <h3>📤 Drop your video here or click to select</h3>
                <input type="file" id="fileInput" accept="video/*" style="display: none;">
                <div class="controls">
                    <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">
                        Choose File
                    </button>
                </div>
            </div>

            <div id="cameraView" class="camera-view">
                <video id="video" autoplay playsinline></video>
                <div class="controls">
                    <button class="btn btn-primary" id="startBtn" onclick="startCamera()">
                        ▶️ Start
                    </button>
                    <button class="btn btn-danger" id="stopBtn" onclick="stopCamera()" disabled>
                        ⏹️ Stop
                    </button>
                </div>
            </div>

            <div id="statusBar" class="status-bar">
                <div id="statusText"></div>
                <div id="confidence" class="confidence"></div>
            </div>

            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>Analyzing video, please wait...</p>
            </div>
        </div>
    </div>

    <audio id="alertSound" preload="auto">
        <source src="data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAADhAC7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7v////////////////////////////////////////////////////////////////AAAAATGF2YzU4LjEzAAAAAAAAAAAAAAAAJAAAAAAAAAAAA4RxGYn0AAAAAAD/+0DEAAAI+AFoAAgAVooA1gAACCQHmFhOiBmZhZmYGZmBhYeJg8PDw8PDw8PDw8PDw8PDw8PDw8PDw8PDw8PDw8PDw8PDw8PDw8P/+0DEKgAKdAFsAAgAWCgDWAAAQAAD//////////////////////////////////////////////////8=" type="audio/mpeg">
    </audio>

    <script>
        let ws = null, videoStream = null, isStreaming = false;
        let canvas = document.createElement('canvas');
        let ctx = canvas.getContext('2d');

        function activateUpload() {
            document.getElementById('uploadArea').classList.add('active');
            document.getElementById('cameraView').classList.remove('active');
            document.getElementById('statusBar').classList.remove('active');
        }

        function activateCamera() {
            document.getElementById('cameraView').classList.add('active');
            document.getElementById('uploadArea').classList.remove('active');
            document.getElementById('statusBar').classList.remove('active');
        }

        function cancelUpload() {
            document.getElementById('uploadArea').classList.remove('active');
            document.getElementById('fileInput').value = '';
            document.getElementById('statusBar').classList.remove('active');
        }

        document.getElementById('fileInput').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const loading = document.getElementById('loading');
            const statusBar = document.getElementById('statusBar');
            
            loading.classList.add('active');
            statusBar.classList.remove('active');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                loading.classList.remove('active');
                showStatus(result);

                if (result.class === 'Violence' && result.confidence > 0.6) {
                    playAlert();
                }
            } catch (error) {
                loading.classList.remove('active');
                alert('Analysis error: ' + error.message);
            }
        });

        async function startCamera() {
            try {
                videoStream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480 } 
                });
                
                const video = document.getElementById('video');
                video.srcObject = videoStream;
                
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                
                ws = new WebSocket('ws://' + window.location.host + '/ws/camera');
                
                ws.onopen = () => {
                    console.log('Connected');
                    isStreaming = true;
                    sendFrames();
                };
                
                ws.onmessage = (event) => {
                    const result = JSON.parse(event.data);
                    showStatus(result);
                    if (result.alert) playAlert();
                };
                
                ws.onclose = () => {
                    console.log('Disconnected');
                    isStreaming = false;
                };
                
            } catch (error) {
                alert('Camera access failed: ' + error.message);
            }
        }

        function stopCamera() {
            isStreaming = false;
            if (ws) { ws.close(); ws = null; }
            if (videoStream) {
                videoStream.getTracks().forEach(track => track.stop());
                videoStream = null;
            }
            document.getElementById('video').srcObject = null;
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('statusBar').classList.remove('active');
        }

        function sendFrames() {
            if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            const video = document.getElementById('video');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            
            const frame = canvas.toDataURL('image/jpeg', 0.8);
            ws.send(JSON.stringify({ frame: frame }));
            
            setTimeout(sendFrames, 500);
        }

        function showStatus(result) {
            const statusBar = document.getElementById('statusBar');
            const statusText = document.getElementById('statusText');
            const confidence = document.getElementById('confidence');
            
            statusBar.classList.add('active');
            statusBar.classList.remove('safe', 'danger');
            
            if (result.class === 'Violence') {
                statusBar.classList.add('danger');
                statusText.textContent = '⚠️ VIOLENCE DETECTED!';
            } else {
                statusBar.classList.add('safe');
                statusText.textContent = '✅ SAFE - No Violence Detected';
            }
            
            confidence.textContent = `Confidence Level: ${(result.confidence * 100).toFixed(1)}%`;
        }

        function playAlert() {
            const audio = document.getElementById('alertSound');
            audio.volume = 0.7;
            audio.play().catch(e => console.log('Audio playback failed:', e));
        }

        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(255, 255, 255, 0.15)';
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = 'rgba(255, 255, 255, 0.03)';
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = 'rgba(255, 255, 255, 0.03)';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('video/')) {
                document.getElementById('fileInput').files = e.dataTransfer.files;
                document.getElementById('fileInput').dispatchEvent(new Event('change'));
            }
        });
    </script>
</body>
</html>"""


INCIDENTS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Incidents Log - Draa</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: white;
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            text-align: center;
            margin-bottom: 35px;
            animation: fadeIn 1s;
        }
        .header h1 {
            font-size: 44px;
            color: #0E6B65;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
            margin-bottom: 12px;
            font-weight: 700;
        }
        .header p { 
            color: rgba(14, 107, 111, 0.98); 
            font-size: 18px;
            font-weight: 500;
        }
        .nav-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #0E6B65;
            padding: 25px 35px;
            border-radius: 20px;
            margin-bottom: 35px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        }
        .btn {
            padding: 14px 30px;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 700;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
            color: #0E6B65;
            box-shadow: 0 5px 15px rgba(255, 255, 255, 0.4);
        }
        .btn-primary:hover {
            box-shadow: 0 8px 25px rgba(255, 255, 255, 0.5);
            transform: translateY(-3px);
        }
        .btn-danger { 
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
            color: #ffffff;
            box-shadow: 0 5px 15px rgba(220, 53, 69, 0.4);
        }
        .btn-danger:hover {
            box-shadow: 0 8px 25px rgba(220, 53, 69, 0.5);
            transform: translateY(-3px);
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-bottom: 35px;
        }
        .stat-card {
            background: #0E6B65;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            text-align: center;
            animation: slideUp 0.6s ease;
            transition: all 0.3s ease;
        }
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.25);
        }
        .stat-card h3 {
            font-size: 40px;
            color: #ffffff;
            margin-bottom: 12px;
            font-weight: 700;
        }
        .stat-card p { 
            color: rgba(255, 255, 255, 0.9); 
            font-size: 15px;
            font-weight: 600;
        }
        .incidents-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        .incident-card {
            background: #0E6B65;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            transition: all 0.4s ease;
            animation: slideUp 0.6s ease;
        }
        .incident-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
        .incident-image {
            width: 100%;
            height: 220px;
            object-fit: cover;
            background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        }
        .incident-content { padding: 25px; }
        .incident-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
        }
        .incident-badge {
            padding: 10px 18px;
            border-radius: 25px;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: linear-gradient(135deg, #D4AF7A 0%, #0E6B65 100%);
            color: white;
        }
        .incident-time { 
            color: #ffffff; 
            font-size: 14px;
            font-weight: 600;
        }
        .incident-details { margin-bottom: 18px; }
        .detail-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 15px;
        }
        .detail-label { 
            color: rgba(255,255,255,0.8);
            font-weight: 500;
        }
        .detail-value { 
            font-weight: 700; 
            color: #ffffff;
        }
        .confidence-bar {
            width: 100%;
            height: 10px;
            background: rgba(255,255,255,0.2);
            border-radius: 5px;
            overflow: hidden;
            margin-top: 12px;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #dc3545 0%, #ff6b6b 100%);
            transition: width 0.5s ease;
        }
        .incident-actions {
            display: flex;
            gap: 12px;
            margin-top: 18px;
        }
        .btn-small {
            flex: 1;
            padding: 10px;
            font-size: 13px;
            border-radius: 10px;
            font-weight: 700;
        }
        .empty-state {
            background: white;
            border-radius: 20px;
            padding: 70px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .empty-state h2 {
            color: #0E6B65;
            font-size: 32px;
            margin-bottom: 18px;
            font-weight: 700;
        }
        .empty-state p { 
            color: #666; 
            font-size: 17px;
            line-height: 1.6;
        }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.85);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            animation: fadeIn 0.3s ease;
        }
        .modal.active { display: flex; }
        .modal-content {
            background: white;
            border-radius: 20px;
            padding: 35px;
            max-width: 900px;
            max-height: 90vh;
            overflow-y: auto;
            position: relative;
            animation: slideUp 0.4s ease;
        }
        .modal-close {
            position: absolute;
            top: 20px;
            right: 20px;
            background: #dc3545;
            color: white;
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 24px;
            line-height: 1;
            transition: all 0.3s ease;
            font-weight: bold;
            z-index: 10;
        }
        .modal-close:hover {
            background: #c82333;
            transform: rotate(90deg);
        }
        .modal-image {
            width: 100%;
            border-radius: 15px;
            margin-bottom: 25px;
            max-height: 450px;
            object-fit: contain;
            background: #f5f5f5;
        }
        .modal-video {
            width: 100%;
            border-radius: 15px;
            margin-bottom: 25px;
            max-height: 550px;
            background: #000;
        }
        .video-badge {
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: 700;
            margin-left: 10px;
        }
        @keyframes fadeIn { 
            from { opacity: 0; } 
            to { opacity: 1; } 
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 768px) {
            .incidents-grid { grid-template-columns: 1fr; }
            .header h1 { font-size: 34px; }
            .stat-card h3 { font-size: 32px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Incidents Log</h1>
            <p>All detected violence cases with recorded videos</p>
        </div>

        <div class="nav-bar">
            <button class="btn btn-primary" onclick="window.location.href='/'">
                🏠 Back to Home
            </button>
            <button class="btn btn-danger" onclick="clearAllIncidents()">
                🗑️ Clear All
            </button>
        </div>

        <div class="stats">
            <div class="stat-card">
                <h3 id="totalIncidents">0</h3>
                <p>Total Incidents</p>
            </div>
            <div class="stat-card">
                <h3 id="todayIncidents">0</h3>
                <p>Today's Incidents</p>
            </div>
            <div class="stat-card">
                <h3 id="avgConfidence">0%</h3>
                <p>Avg Confidence</p>
            </div>
            <div class="stat-card">
                <h3 id="lastIncident">-</h3>
                <p>Last Incident</p>
            </div>
        </div>

        <div id="incidentsContainer" class="incidents-grid"></div>

        <div id="emptyState" class="empty-state" style="display: none;">
            <h2>No Incidents Recorded</h2>
            <p>Start detecting violence to see results and alerts here</p>
            <button class="btn btn-primary" onclick="window.location.href='/'" style="margin-top: 25px;">
                Start Detection
            </button>
        </div>
    </div>

    <div id="modal" class="modal" onclick="closeModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <button class="modal-close" onclick="closeModal()">×</button>
            <div id="modalBody"></div>
        </div>
    </div>

    <script>
        let incidents = [];

        async function loadIncidents() {
            try {
                const response = await fetch('/api/incidents');
                incidents = await response.json();
                console.log('Loaded incidents:', incidents);
                renderIncidents();
                updateStats();
            } catch (error) {
                console.error('Error loading incidents:', error);
            }
        }

        function renderIncidents() {
            const container = document.getElementById('incidentsContainer');
            const emptyState = document.getElementById('emptyState');

            if (incidents.length === 0) {
                container.style.display = 'none';
                emptyState.style.display = 'block';
                return;
            }

            container.style.display = 'grid';
            emptyState.style.display = 'none';

            container.innerHTML = incidents.map(incident => {
                const thumbnailPath = incident.thumbnail ? incident.thumbnail.split('/').pop() : '';
                const hasVideo = incident.video ? true : false;
                
                return `
                <div class="incident-card">
                    <img src="/incidents/${thumbnailPath}" 
                         alt="Incident" 
                         class="incident-image" 
                         onerror="this.outerHTML='<div class=\\'incident-image\\' style=\\'display:flex;align-items:center;justify-content:center;color:#999;font-size:16px;font-weight:600\\'>No Image Available</div>'">
                    <div class="incident-content">
                        <div class="incident-header">
                            <span class="incident-badge">
                                ${incident.type === 'camera' ? '📹 Live' : '📤 Upload'}
                                ${hasVideo ? '<span class="video-badge">🎥 VIDEO</span>' : ''}
                            </span>
                            <span class="incident-time">${incident.time || 'N/A'}</span>
                        </div>
                        
                        <div class="incident-details">
                            <div class="detail-row">
                                <span class="detail-label">Date:</span>
                                <span class="detail-value">${incident.date || 'N/A'}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Confidence:</span>
                                <span class="detail-value">${(incident.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${incident.confidence * 100}%"></div>
                            </div>
                        </div>

                        <div class="incident-actions">
                            <button class="btn btn-primary btn-small" onclick='viewDetails(${JSON.stringify(incident).replace(/'/g, "&#39;")})'>
                                👁️ View
                            </button>
                            ${hasVideo ? `
                                <button class="btn btn-primary btn-small" onclick="downloadVideo('${incident.video}')">
                                    📥 Video
                                </button>
                            ` : ''}
                            <button class="btn btn-danger btn-small" onclick="deleteIncident('${incident.id}')">
                                🗑️
                            </button>
                        </div>
                    </div>
                </div>
            `}).join('');
        }

        function updateStats() {
            const today = new Date().toISOString().split('T')[0];
            const todayIncidents = incidents.filter(i => i.date === today);
            const avgConf = incidents.length > 0 
                ? incidents.reduce((sum, i) => sum + i.confidence, 0) / incidents.length 
                : 0;

            document.getElementById('totalIncidents').textContent = incidents.length;
            document.getElementById('todayIncidents').textContent = todayIncidents.length;
            document.getElementById('avgConfidence').textContent = (avgConf * 100).toFixed(1) + '%';
            document.getElementById('lastIncident').textContent = 
                incidents.length > 0 ? incidents[0].time : '-';
        }

        function viewDetails(incident) {
            const modal = document.getElementById('modal');
            const modalBody = document.getElementById('modalBody');

            const thumbnailPath = incident.thumbnail ? incident.thumbnail.split('/').pop() : '';
            const videoPath = incident.video ? incident.video.split('/').pop() : null;

            modalBody.innerHTML = `
                <h2 style="color: #0E6B65; margin-bottom: 25px; font-size: 26px; font-weight: 700;">
                    Incident Details
                </h2>
                
                ${videoPath ? `
                    <video controls class="modal-video" style="width: 100%; max-height: 500px; border-radius: 15px; margin-bottom: 25px;">
                        <source src="/uploads/${videoPath}" type="video/mp4">
                        Your browser does not support video playback.
                    </video>
                ` : thumbnailPath ? `
                    <img src="/incidents/${thumbnailPath}" 
                         alt="Incident" 
                         class="modal-image"
                         onerror="this.outerHTML='<div style=\\'text-align:center;padding:50px;color:#999;font-size:16px;font-weight:600\\'>Image not available</div>'">
                ` : ''}
                
                <div style="background: #f8f9fa; padding: 25px; border-radius: 15px; margin-bottom: 25px;">
                    <div class="detail-row" style="margin-bottom: 18px;">
                        <span class="detail-label">Type:</span>
                        <span class="detail-value">${incident.type === 'camera' ? '📹 Live Camera Detection' : '📤 Video Upload'}</span>
                    </div>
                    <div class="detail-row" style="margin-bottom: 18px;">
                        <span class="detail-label">Date:</span>
                        <span class="detail-value">${incident.date}</span>
                    </div>
                    <div class="detail-row" style="margin-bottom: 18px;">
                        <span class="detail-label">Time:</span>
                        <span class="detail-value">${incident.time}</span>
                    </div>
                    <div class="detail-row" style="margin-bottom: 18px;">
                        <span class="detail-label">Confidence:</span>
                        <span class="detail-value">${(incident.confidence * 100).toFixed(2)}%</span>
                    </div>
                    <div class="detail-row" style="margin-bottom: 18px;">
                        <span class="detail-label">Video Available:</span>
                        <span class="detail-value">${videoPath ? '✅ Yes' : '❌ No'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">ID:</span>
                        <span class="detail-value" style="font-size: 13px; font-family: monospace;">${incident.id}</span>
                    </div>
                </div>

                <div style="display: flex; gap: 15px;">
                    ${videoPath ? `
                        <button class="btn btn-primary" onclick="downloadVideo('${incident.video}')" style="flex: 1;">
                            📥 Download Video
                        </button>
                    ` : ''}
                    <button class="btn btn-danger" onclick="deleteIncident('${incident.id}'); closeModal();" style="flex: 1;">
                        🗑️ Delete Incident
                    </button>
                </div>
            `;

            modal.classList.add('active');
        }

        function closeModal(event) {
            if (event && event.target !== event.currentTarget) return;
            document.getElementById('modal').classList.remove('active');
        }

        async function deleteIncident(id) {
            if (!confirm('Are you sure you want to delete this incident?')) return;

            try {
                const response = await fetch(`/api/incidents/${id}`, { method: 'DELETE' });
                if (response.ok) {
                    await loadIncidents();
                } else {
                    alert('Error deleting incident');
                }
            } catch (error) {
                alert('Error deleting incident: ' + error.message);
            }
        }

        async function clearAllIncidents() {
            if (!confirm('⚠️ Are you sure you want to delete ALL incidents? This action cannot be undone.')) return;

            try {
                for (const incident of incidents) {
                    await fetch(`/api/incidents/${id}`, { method: 'DELETE' });
                }
                await loadIncidents();
            } catch (error) {
                alert('Error clearing incidents: ' + error.message);
            }
        }

        function downloadVideo(videoUrl) {
            const filename = videoUrl.split('/').pop();
            const a = document.createElement('a');
            a.href = '/uploads/' + filename;
            a.download = filename;
            a.target = '_blank';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }

        // Auto-refresh every 10 seconds
        loadIncidents();
        setInterval(loadIncidents, 3000);
    </script>
</body>
</html>"""

# =============== API ROUTES ===============

@app.get("/", response_class=HTMLResponse)
async def home():
    return INDEX_HTML

@app.get("/incidents", response_class=HTMLResponse)
async def incidents_page():
    return INCIDENTS_HTML

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = f"uploads/{filename}"
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = predict_video(filepath)
        
        if result['class'] == 'Violence' and result['confidence'] > 0.6:
            cap = cv2.VideoCapture(filepath)
            ret, frame = cap.read()
            if ret:
                thumbnail_name = f"thumb_{timestamp}.jpg"
                thumbnail_path = f"incidents/{thumbnail_name}"
                cv2.imwrite(thumbnail_path, frame)
            cap.release()
            
            incident = {
                "id": timestamp,
                "type": "upload",
                "timestamp": datetime.now().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "confidence": result['confidence'],
                "thumbnail": thumbnail_path,
                "video": filepath
            }
            save_incident(incident)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/incidents")
async def get_incidents():
    incidents = load_incidents()
    return JSONResponse(content=incidents)

@app.delete("/api/incidents/{incident_id}")
async def delete_incident(incident_id: str):
    incidents = load_incidents()
    
    # Delete associated files
    for incident in incidents:
        if incident['id'] == incident_id:
            # Delete thumbnail
            if incident.get('thumbnail') and os.path.exists(incident['thumbnail']):
                try:
                    os.remove(incident['thumbnail'])
                except Exception as e:
                    print(f"Error deleting thumbnail: {e}")
            # Delete video
            if incident.get('video') and os.path.exists(incident['video']):
                try:
                    os.remove(incident['video'])
                except Exception as e:
                    print(f"Error deleting video: {e}")
            break
    
    incidents = [i for i in incidents if i['id'] != incident_id]
    with open(INCIDENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(incidents, f, indent=2, ensure_ascii=False)
    return JSONResponse(content={"success": True})

@app.get("/incidents/{filename}")
async def get_incident_file(filename: str):
    file_path = f"incidents/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(content={"error": "File not found"}, status_code=404)

@app.get("/uploads/{filename}")
async def get_upload_file(filename: str):
    file_path = f"uploads/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(content={"error": "File not found"}, status_code=404)



@app.websocket("/ws/camera")
async def camera_feed(websocket: WebSocket):
    await websocket.accept()
    temporal_checker = TemporalConsistencyChecker()
    video_recorder = VideoRecorder(pre_buffer_seconds=5, post_buffer_seconds=5, frame_rate=10)
    
    frame_buffer = []
    last_violence_time = 0
    frame_count = 0
    violence_detected = False
    current_incident_timestamp = None
    current_max_confidence = 0.0
    
    print("=" * 60)
    print("🎥 Camera connection established")
    print("=" * 60)
    
    try:
        while True:
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            img_data = base64.b64decode(frame_data['frame'].split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            frame_count += 1
            
            # Always add frame to recorder's buffer
            video_recorder.add_frame(frame)
            
            # Preprocess frame for model
            frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
            frame_rgb = frame_resized[:, :, [2, 1, 0]]  # BGR to RGB
            frame_normalized = frame_rgb / 255.0  # Normalize
            frame_buffer.append(frame_normalized)
            
            if len(frame_buffer) > SEQUENCE_LENGTH * 2:
                frame_buffer = frame_buffer[-SEQUENCE_LENGTH:]
            
            # Make prediction
            if len(frame_buffer) >= SEQUENCE_LENGTH:
                sequence = np.array(frame_buffer[-SEQUENCE_LENGTH:])
                sequence = np.expand_dims(sequence, axis=0)
                
                # Get raw prediction (sigmoid output)
                raw_prediction = model.predict(sequence, verbose=0)[0][0]
                
                # Smooth prediction (pass as array for temporal checker)
                smoothed_pred = temporal_checker.update(np.array([1-raw_prediction, raw_prediction]))
                
                # Determine violence
                violence_prob = smoothed_pred[1]  # Probability of Violence
                is_violence = violence_prob > 0.5
                confidence = violence_prob if is_violence else smoothed_pred[0]
                class_name = 'Violence' if is_violence else 'Normal'


                # Handle recording logic
                if is_violence:
                    violence_detected = True
                    last_violence_time = time.time()
                    # Track max confidence during incident
                    if confidence > current_max_confidence:
                        current_max_confidence = float(confidence)
                    # Start recording if not already recording
                    if not video_recorder.is_recording:
                        current_incident_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        video_path = video_recorder.start_recording(current_incident_timestamp)
                        
                        # Save thumbnail immediately
                        thumbnail_name = f"camera_{timestamp}.jpg"
                        thumbnail_path = f"incidents/{thumbnail_name}"
                        cv2.imwrite(thumbnail_path, frame)
                        print(f"📸 Thumbnail saved: {thumbnail_path}")

                # Add frame to recording if active
                if video_recorder.is_recording:
                    video_recorder.add_recording_frame(frame)
                
                # Check if violence has ended
                if violence_detected and not is_violence:
                    if time.time() - last_violence_time > 1.0:  # 1 second of no violence
                        video_recorder.mark_fight_ended()
                        violence_detected = False
                
                # Check if recording should stop
                if video_recorder.should_stop_recording():
                    final_video_path, timestamp = video_recorder.stop_recording()
                    
                    # Save incident with video
                    if final_video_path and current_incident_timestamp:
                        thumbnail_path = f"incidents/camera_{current_incident_timestamp}.jpg"
                        
                        # Verify files exist
                        video_exists = os.path.exists(final_video_path)
                        thumbnail_exists = os.path.exists(thumbnail_path)
                        
                        print(f"💾 Saving incident...")
                        print(f"   Video: {final_video_path} (exists: {video_exists})")
                        print(f"   Thumbnail: {thumbnail_path} (exists: {thumbnail_exists})")
                        
                        incident = {
                            "id": current_incident_timestamp,
                            "type": "camera",
                            "timestamp": datetime.now().isoformat(),
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "confidence": current_max_confidence,
                            "thumbnail": thumbnail_path if thumbnail_exists else None,
                            "video": final_video_path if video_exists else None
                        }
                        save_incident(incident)
                        print(f"🚨 CAMERA INCIDENT SAVED!")
                        print(f"   ID: {current_incident_timestamp}")
                        print(f"   Confidence: {current_max_confidence:.2%}")
                        print(f"   Video Path: {final_video_path}")
                    
                    # Reset for next incident
                    current_incident_timestamp = None
                    current_max_confidence = 0.0
                
                # Send result to frontend
                result = {
                    'class': class_name,
                    'confidence': float(confidence),
                    'alert': bool(is_violence),
                    'is_recording': video_recorder.is_recording,
                    'frame_count': int(frame_count)
                }

                await websocket.send_json(result)
            else:
                await websocket.send_json({
                    'class': 'Loading',
                    'confidence': 0.0,
                    'alert': False,
                    'is_recording': False,
                    'frames_buffered': len(frame_buffer),
                    'frames_needed': SEQUENCE_LENGTH
                })
    
    except WebSocketDisconnect:
        print("\n❌ Camera disconnected")
        if video_recorder.is_recording:
            video_recorder.stop_recording()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        if video_recorder.is_recording:
            video_recorder.stop_recording()

# =============== RUN SERVER ===============
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Draa - Smart Violence Detection System")
    print("=" * 60)
    print("📍 URL: http://localhost:8000")
    print("📊 Incidents: http://localhost:8000/incidents")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)