# Facial Heart Rate Detection API

A production-ready REST API for facial heart rate detection using image processing. This backend service processes uploaded images to detect heart rate using remote photoplethysmography (rPPG) techniques.

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Build and run the API
docker-compose up -d

# Or build manually
docker build -t facial-heart-rate-api .
docker run -d -p 8000:8000 --name facial-heart-rate-api facial-heart-rate-api
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python server.py
```

### Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🏗️ Architecture

### Image Processing Flow

```
Image Upload → Face Detection → ROI Extraction → Signal Processing → Heart Rate Calculation
     ↓              ↓               ↓                ↓                    ↓
Base64 Input → MediaPipe → Forehead Region → rPPG Analysis → BPM Output
```

### Core Technologies

- **FastAPI**: Modern Python web framework
- **MediaPipe**: Face landmark detection
- **OpenCV**: Image processing
- **SciPy**: Signal filtering and frequency analysis
- **NumPy**: Numerical computations
- **Docker**: Containerization

## 📋 API Endpoints

### Session Management
- `POST /create-session` - Create a new analysis session
- `GET /session/{session_id}/status` - Get session status and metrics
- `DELETE /session/{session_id}` - Delete a specific session
- `GET /sessions` - List all active sessions

### Image Analysis
- `POST /analyze-image` - Analyze a single image for heart rate
- `POST /analyze-batch` - Analyze multiple images in batch

### System Monitoring
- `GET /health` - Health check endpoint
- `GET /status` - Comprehensive system status
- `GET /metrics` - Performance metrics and statistics

## 🔬 How It Works

### Heart Rate Detection Process

1. **Face Detection**: MediaPipe identifies facial landmarks
2. **ROI Extraction**: Extract forehead region (optimal for rPPG)
3. **Signal Processing**:
   - Extract green channel values (most sensitive to blood volume changes)
   - Apply bandpass filter (42-240 BPM range)
   - Detrend and normalize the signal
4. **Frequency Analysis**: Use Welch's method for power spectral density
5. **BPM Calculation**: Identify peak frequency and convert to beats per minute

### Session-Based Analysis

- Each session maintains a buffer of signal data
- Multiple images improve accuracy through temporal analysis
- Sessions automatically expire after 5 minutes of inactivity
- Real-time signal quality assessment

## 🔧 Configuration

### Environment Variables

```bash
PYTHONUNBUFFERED=1      # Disable Python output buffering
PYTHONPATH=/app         # Set Python path
PORT=8000              # API port (default: 8000)
LOG_LEVEL=info         # Logging level
```

### System Limits

- **Maximum Sessions**: 100 concurrent sessions
- **Session Timeout**: 5 minutes of inactivity
- **Buffer Size**: 200 frames per session
- **Heart Rate Range**: 42-240 BPM

## 🧪 API Usage Examples

### Create a Session

```bash
curl -X POST http://localhost:8000/create-session
```

Response:
```json
{
  "session_id": "uuid-string",
  "message": "Session created successfully"
}
```

### Analyze an Image

```bash
curl -X POST http://localhost:8000/analyze-image \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "image_data": "base64-encoded-image-data"
  }'
```

Response:
```json
{
  "heart_rate": 72,
  "confidence": "high",
  "face_detected": true,
  "session_id": "your-session-id",
  "message": "Analysis completed successfully"
}
```

### Check System Health

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-05-26T15:30:00.000000",
  "version": "2.1.0",
  "uptime_seconds": 1234.56
}
```

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build the image
docker build -t facial-heart-rate-api:latest .

# Run with production settings
docker run -d \
  --name facial-heart-rate-api \
  -p 8000:8000 \
  --restart unless-stopped \
  -v ./data:/app/data \
  facial-heart-rate-api:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  heart-rate-api:
    build: .
    container_name: facial-heart-rate-api
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=info
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## 📊 Performance & Scaling

### System Requirements

- **CPU**: 2+ cores recommended
- **RAM**: 2GB minimum, 4GB recommended  
- **Storage**: 1GB for Docker images + data
- **Network**: Stable connection for API requests

### Optimization Tips

1. **Image Quality**: Use well-lit, clear images (400x400px or larger)
2. **Batch Processing**: Use `/analyze-batch` for multiple images
3. **Session Reuse**: Maintain sessions for continuous monitoring
4. **Resource Monitoring**: Use `/metrics` endpoint for monitoring

### Cloud Deployment

Suitable for deployment on:
- **AWS ECS/Fargate**
- **Google Cloud Run**
- **Azure Container Instances**
- **Kubernetes clusters**
- **DigitalOcean App Platform**

## 🔒 Security Considerations

### API Security

- Input validation with Pydantic models
- CORS configuration for allowed origins
- Rate limiting (implement as needed)
- No external dependencies on camera hardware

### Container Security

- Non-root user in container
- Minimal base image (python:3.11-slim)
- Multi-stage build for reduced attack surface
- Health checks for monitoring

## 🔍 Monitoring & Observability

### Built-in Monitoring

- **Health Endpoint**: `/health` - Basic health status
- **Metrics Endpoint**: `/metrics` - Detailed performance metrics
- **Status Endpoint**: `/status` - Comprehensive system status

### Integration Options

- Prometheus + Grafana
- DataDog
- New Relic
- AWS CloudWatch
- Custom monitoring solutions

## 🐛 Troubleshooting

### Common Issues

1. **Poor Heart Rate Accuracy**
   - Ensure good lighting conditions
   - Use high-quality images (>400x400px)
   - Send multiple images for better temporal analysis

2. **Session Limit Reached**
   - Check active sessions: `GET /sessions`
   - Clean up unused sessions: `DELETE /session/{id}`
   - Sessions auto-expire after 5 minutes

3. **Face Detection Failed**
   - Verify image contains a clear frontal face
   - Check image encoding (base64) is correct
   - Ensure adequate lighting and resolution

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=debug
python server.py
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions welcome! Please submit pull requests for improvements.

---

**⚠️ Disclaimer**: This system is for educational and research purposes only. Not intended for medical diagnosis. Always consult healthcare professionals for medical advice.

### Session-Based Analysis

- Each analysis session maintains a buffer of signal data
- Multiple images improve accuracy through temporal analysis
- Sessions automatically expire after 5 minutes of inactivity
- Real-time signal quality assessment

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build the production image
docker build -t facial-heart-rate-api:latest .

# Run with production settings
docker run -d \
  --name facial-heart-rate-api \
  -p 8000:8000 \
  --restart unless-stopped \
  facial-heart-rate-api:latest
```

### Docker Compose (Full Stack)

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - api
    restart: unless-stopped
```

## 🔧 Configuration

### Backend Configuration

Environment variables:
- `PYTHONUNBUFFERED=1` - Disable Python output buffering
- `PYTHONPATH=/app` - Set Python path
- `PORT=8000` - API port (default: 8000)

### Frontend Configuration

Create `.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production, update the API URL to your backend service.

## 📊 Performance Considerations

### System Requirements

- **CPU**: 2+ cores recommended
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 1GB for Docker images
- **Network**: Stable internet for real-time processing

### Optimization Tips

1. **Image Quality**: Use good lighting and clear images
2. **Batch Processing**: Send multiple images for better accuracy
3. **Session Management**: Reuse sessions for continuous monitoring
4. **Caching**: Frontend caches recent results
5. **Error Handling**: Graceful degradation on processing errors

## 🚀 Scaling and Production

### Cloud Deployment Options

1. **AWS ECS/Fargate**
2. **Google Cloud Run**
3. **Azure Container Instances**
4. **Kubernetes clusters**
5. **DigitalOcean App Platform**

### Load Balancing

For high traffic:
- Deploy multiple API instances
- Use nginx or cloud load balancer
- Implement session affinity if needed
- Monitor resource usage

### Monitoring

Built-in endpoints for monitoring:
- `/health` - Health status
- `/metrics` - Performance metrics
- `/status` - System status

Integrate with:
- Prometheus + Grafana
- DataDog
- New Relic
- AWS CloudWatch

## 🔒 Security

### API Security

- CORS configured for specific domains
- Input validation with Pydantic
- Rate limiting (implement as needed)
- No camera access required on server

### Docker Security

- Non-root user in container
- Minimal base image (python:3.11-slim)
- Multi-stage build for smaller attack surface
- Health checks for container monitoring

## 🧪 Testing

### API Testing

```bash
# Health check
curl http://localhost:8000/health

# Create session
curl -X POST http://localhost:8000/create-session

# Upload test image (base64 encoded)
curl -X POST http://localhost:8000/analyze-image \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "base64_encoded_image_data",
    "session_id": "your_session_id"
  }'
```

### Frontend Testing

```bash
cd frontend
npm run build
npm run start
```

## 📝 Development

### Local Development

1. **Backend**:
   ```bash
   pip install -r requirements.txt
   python server_image_based.py
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

### Adding Features

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Test thoroughly
5. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

1. **Camera Permission Denied**
   - Frontend issue: Check browser permissions
   - Use HTTPS in production for camera access

2. **API Connection Failed**
   - Check if backend is running on correct port
   - Verify CORS settings
   - Check network connectivity

3. **Poor Heart Rate Accuracy**
   - Ensure good lighting
   - Keep face still during capture
   - Use multiple images for better results

4. **Docker Build Issues**
   - Check Docker daemon is running
   - Verify system requirements
   - Clear Docker cache if needed

### Debug Mode

Enable debug logging:
```bash
# Backend
PYTHONPATH=/app python -c "import logging; logging.basicConfig(level=logging.DEBUG)"

# Frontend
NEXT_PUBLIC_DEBUG=true npm run dev
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: Built-in API docs at `/docs`
- **Examples**: See frontend code for usage examples

---

**Note**: This system is for educational and research purposes. Not intended for medical diagnosis. Always consult healthcare professionals for medical advice.
