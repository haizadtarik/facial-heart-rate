# ❤️ Facial Heart Rate Detection API

A production-ready FastAPI backend service that uses facial detection and photoplethysmography (PPG) to estimate heart rate from webcam video feed. Designed as a standalone API service that can integrate with any frontend framework (React, Next.js, Vue.js, etc.) and deploy seamlessly to AWS cloud infrastructure.

## 🌟 Features

### Core API Features
- **🔄 RESTful API** - Clean REST endpoints with OpenAPI documentation
- **📹 Real-time Video Processing** - Live camera feed with heart rate overlay
- **❤️ Accurate Heart Rate Detection** - Advanced signal processing with confidence scoring
- **📊 System Monitoring** - Health checks, metrics, and status endpoints
- **🔐 Production Ready** - Proper error handling, logging, and security headers
- **📋 Interactive Documentation** - Auto-generated Swagger UI and ReDoc

### Technical Features
- **🐳 Docker Support** - Containerized for easy deployment
- **☁️ AWS Ready** - CloudFormation templates and deployment scripts
- **🔄 Auto-scaling** - ECS Fargate with auto-scaling configuration
- **📊 Monitoring** - CloudWatch logs and metrics
- **🔒 Security** - Non-root containers, security groups, and proper IAM roles
- **⚡ Performance** - Optimized processing with configurable parameters

### Integration Features
- **🌐 CORS Enabled** - Ready for frontend integration
- **📱 Mobile Friendly** - Works with React Native and mobile apps
- **🔌 Webhook Support** - Real-time data streaming capabilities
- **📈 Scalable Architecture** - Microservice design pattern

## 🛠️ Technology Stack

- **Backend Framework**: FastAPI with Pydantic models
- **Computer Vision**: OpenCV + MediaPipe for face detection
- **Signal Processing**: SciPy for heart rate analysis
- **Containerization**: Docker with multi-stage builds
- **Cloud Platform**: AWS ECS Fargate with Application Load Balancer
- **Infrastructure**: CloudFormation for Infrastructure as Code
- **Monitoring**: CloudWatch for logs and metrics

## 📋 Requirements

- **Python**: 3.8+
- **Camera**: Webcam or USB camera
- **Lighting**: Good lighting conditions for face detection
- **OS**: macOS, Windows, or Linux
- **Memory**: 2GB+ RAM recommended
- **Docker**: For containerized deployment
- **AWS CLI**: For cloud deployment

## 🚀 Quick Start

### Local Development

1. **Setup Development Environment**:
```bash
# Clone the repository
git clone <repository-url>
cd facial-heart-rate

# Run the setup script
./setup-dev.sh
```

2. **Start Development Server**:
```bash
./start.sh
```

3. **Test the API**:
```bash
# Health check
curl http://localhost:8000/health

# Get heart rate
curl http://localhost:8000/bpm

# View API documentation
open http://localhost:8000/docs
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run individual container
docker build -t heart-rate-api .
docker run -p 8000:8000 --device=/dev/video0 heart-rate-api
```

## 📱 Frontend Integration

### Next.js with Tailwind CSS Example

```typescript
// components/HeartRateMonitor.tsx
import { useState, useEffect } from 'react';

interface HeartRateData {
  bpm: number;
  confidence: string;
  face_detected: boolean;
  signal_quality: string;
}

export default function HeartRateMonitor() {
  const [heartRate, setHeartRate] = useState<HeartRateData | null>(null);
  const [videoSrc, setVideoSrc] = useState<string>('');

  useEffect(() => {
    // Set video feed source
    setVideoSrc('http://localhost:8000/video_feed');

    // Poll heart rate data
    const interval = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8000/bpm');
        const data = await response.json();
        setHeartRate(data);
      } catch (error) {
        console.error('Failed to fetch heart rate:', error);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold text-gray-800 mb-8">
        Heart Rate Monitor
      </h1>
      
      {/* Video Feed */}
      <div className="relative mb-6">
        <img 
          src={videoSrc} 
          alt="Live Camera Feed"
          className="rounded-lg shadow-lg w-640 h-480"
        />
      </div>

      {/* Heart Rate Display */}
      {heartRate && (
        <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
          <div className="text-center">
            <div className="text-4xl font-bold text-red-500 mb-2">
              {heartRate.bpm} BPM
            </div>
            <div className={`text-sm font-medium mb-4 ${
              heartRate.confidence === 'high' ? 'text-green-600' : 
              heartRate.confidence === 'medium' ? 'text-yellow-600' : 
              'text-red-600'
            }`}>
              Confidence: {heartRate.confidence}
            </div>
            <div className={`text-sm ${
              heartRate.face_detected ? 'text-green-600' : 'text-red-600'
            }`}>
              Face: {heartRate.face_detected ? 'Detected' : 'Not Found'}
            </div>
            <div className={`text-sm ${
              heartRate.signal_quality === 'good' ? 'text-green-600' : 
              heartRate.signal_quality === 'fair' ? 'text-yellow-600' : 
              'text-red-600'
            }`}>
              Signal: {heartRate.signal_quality}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
```

### React Hook for Heart Rate Data

```typescript
// hooks/useHeartRate.ts
import { useState, useEffect, useCallback } from 'react';

interface HeartRateData {
  bpm: number;
  confidence: string;
  face_detected: boolean;
  signal_quality: string;
  timestamp: number;
}

export function useHeartRate(apiUrl: string = 'http://localhost:8000') {
  const [data, setData] = useState<HeartRateData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHeartRate = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/bpm`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const heartRateData = await response.json();
      setData(heartRateData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }, [apiUrl]);

  useEffect(() => {
    fetchHeartRate();
    const interval = setInterval(fetchHeartRate, 2000);
    return () => clearInterval(interval);
  }, [fetchHeartRate]);

  return { data, isLoading, error, refetch: fetchHeartRate };
}
```

## 🔧 API Endpoints

### Core Endpoints

| Method | Endpoint | Description | Response Model |
|--------|----------|-------------|----------------|
| GET | `/` | Root health check | HealthCheckResponse |
| GET | `/health` | Detailed health status | HealthCheckResponse |
| GET | `/bpm` | Current heart rate data | HeartRateResponse |
| GET | `/status` | System status | SystemStatusResponse |
| GET | `/current_frame` | Current camera frame | FrameResponse |
| GET | `/video_feed` | Live video stream | MJPEG Stream |
| GET | `/metrics` | System metrics | JSON |

### Response Models

**HeartRateResponse**:
```json
{
  "bpm": 72,
  "confidence": "high",
  "message": "Face detected",
  "face_detected": true,
  "buffer_fill": "180/200",
  "timestamp": 1640995200.0,
  "signal_quality": "good"
}
```

**SystemStatusResponse**:
```json
{
  "camera_active": true,
  "face_detected": true,
  "buffer_size": 180,
  "max_buffer_size": 200,
  "time_since_last_update": 1.2,
  "signal_quality": "good",
  "api_version": "2.0.0",
  "system_health": "healthy"
}
```

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## ⚙️ Configuration

### Environment Variables

```bash
# Application Settings
ENVIRONMENT=production              # development, staging, production
LOG_LEVEL=info                     # debug, info, warning, error
DEBUG=false                        # true, false

# Camera Configuration
CAMERA_WIDTH=640                   # Camera resolution width
CAMERA_HEIGHT=480                  # Camera resolution height
CAMERA_FPS=20                      # Target frames per second

# Heart Rate Detection
BUFFER_SIZE=200                    # Signal buffer size (~10s at 20 FPS)
MIN_HZ=0.7                        # Minimum heart rate frequency (42 BPM)
MAX_HZ=4.0                        # Maximum heart rate frequency (240 BPM)
SAMPLING_RATE=20                   # Target processing rate
CONFIDENCE_THRESHOLD=0.6           # Face detection confidence threshold

# Health Monitoring
MAX_NO_UPDATE_SECONDS=10           # Seconds before marking as unhealthy
MAX_POOR_SIGNAL_SECONDS=30         # Seconds of poor signal before alert

# API Settings
API_HOST=0.0.0.0                  # Bind address
API_PORT=8000                     # Port number
CORS_ORIGINS=*                    # CORS allowed origins
```

### Development vs Production

**Development** (`dev-start.sh`):
- Hot reload enabled
- Debug logging
- CORS wildcard
- Single worker

**Production** (Docker/AWS):
- No reload
- Info logging
- Specific CORS origins
- Health checks
- Resource limits

## 🐳 Docker Configuration

### Dockerfile Features

- **Multi-stage build** for optimized image size
- **Non-root user** for security
- **Health checks** for container monitoring
- **Optimized dependencies** for faster builds
- **Proper signal handling** for graceful shutdown

### Docker Compose Features

- **Service definition** with proper resource limits
- **Volume mounts** for camera access and data persistence
- **Health checks** and restart policies
- **Network configuration** for service communication
- **Environment variable** management

## ☁️ AWS Deployment

### Architecture

```
Internet → ALB → ECS Fargate → ECR
                    ↓
               CloudWatch Logs
                    ↓
               Auto Scaling
```

### CloudFormation Stack

The deployment creates:
- **VPC** with public subnets
- **Application Load Balancer** with health checks
- **ECS Fargate cluster** with auto-scaling
- **ECR repository** for container images
- **IAM roles** with minimal permissions
- **CloudWatch logs** for monitoring
- **Security groups** with proper access control

### Deployment Commands

```bash
# One-time setup
aws configure

# Deploy everything
./deploy-aws.sh

# Manual deployment steps
aws ecr create-repository --repository-name heart-rate-api
docker build -t heart-rate-api .
docker tag heart-rate-api:latest $ECR_URI:latest
docker push $ECR_URI:latest
aws cloudformation deploy --template-file aws-cloudformation.yml --stack-name heart-rate-api
```

### Monitoring and Scaling

- **Auto-scaling**: Based on CPU utilization (70% threshold)
- **Health checks**: Every 30 seconds with 3 retry attempts
- **Logs**: Centralized in CloudWatch with 30-day retention
- **Metrics**: Custom metrics for heart rate processing
- **Alerts**: CloudWatch alarms for service health

## 🧪 Testing

### Local Testing

```bash
# API health check
curl http://localhost:8000/health

# Get current heart rate
curl http://localhost:8000/bpm

# System status
curl http://localhost:8000/status

# Stream video feed
curl http://localhost:8000/video_feed

# Test with different lighting conditions
# Test with multiple faces
# Test with no face visible
```

### Load Testing

```bash
# Install Apache Bench
brew install httpd  # macOS

# Test API endpoints
ab -n 1000 -c 10 http://localhost:8000/health
ab -n 100 -c 5 http://localhost:8000/bpm
```

### Integration Testing

```python
import requests
import time

def test_api_integration():
    base_url = "http://localhost:8000"
    
    # Test health check
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    
    # Test heart rate endpoint
    response = requests.get(f"{base_url}/bpm")
    assert response.status_code in [200, 503]  # 503 if no camera
    
    # Test video feed
    response = requests.get(f"{base_url}/video_feed", stream=True)
    assert response.status_code in [200, 503]
    
    print("✅ All integration tests passed")

if __name__ == "__main__":
    test_api_integration()
```

## 🔧 Troubleshooting

### Common Issues

**Camera Access Issues**:
```bash
# macOS: Grant camera permissions in System Preferences
# Linux: Check device permissions
ls -la /dev/video*
sudo chmod 666 /dev/video0

# Docker: Ensure device mapping
docker run --device=/dev/video0 heart-rate-api
```

**Face Detection Problems**:
- Ensure good lighting (avoid backlighting)
- Position face 2-3 feet from camera
- Keep face fully visible in frame
- Remove glasses if possible
- Minimize head movement

**Performance Issues**:
```bash
# Monitor CPU usage
docker stats

# Check logs
docker logs <container-id>

# Adjust camera resolution
export CAMERA_WIDTH=320
export CAMERA_HEIGHT=240
```

**AWS Deployment Issues**:
```bash
# Check ECS service status
aws ecs describe-services --cluster heart-rate-cluster --services heart-rate-service

# View logs
aws logs describe-log-groups
aws logs get-log-events --log-group-name /ecs/heart-rate-api

# Check load balancer health
aws elbv2 describe-target-health --target-group-arn <arn>
```

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 503 | Camera not active | Check camera connection and permissions |
| 500 | Processing error | Check logs for specific error details |
| 429 | Rate limited | Reduce request frequency |
| 404 | Endpoint not found | Check API documentation |

## 📊 Performance Optimization

### Backend Optimizations

- **Signal Processing**: Optimized buffer management and filtering
- **Memory Usage**: Efficient NumPy array operations
- **CPU Usage**: Configurable processing rates and image resolution
- **Threading**: Non-blocking camera processing thread

### Frontend Integration Tips

- **Polling Strategy**: Use 2-3 second intervals for heart rate data
- **Caching**: Cache video feed to reduce bandwidth
- **Error Handling**: Implement exponential backoff for failed requests
- **UI/UX**: Show loading states and connection status

### Production Recommendations

- **Resource Limits**: Set appropriate CPU and memory limits
- **Load Balancing**: Use multiple instances for high availability
- **CDN**: Use CloudFront for global distribution
- **Monitoring**: Set up alerts for service health and performance

## 🔒 Security Considerations

### API Security

- **CORS**: Configure specific origins in production
- **Rate Limiting**: Implement request rate limiting
- **Input Validation**: All inputs validated with Pydantic
- **Error Handling**: No sensitive information in error responses

### Container Security

- **Non-root User**: Containers run as non-privileged user
- **Minimal Base Image**: Use slim Python image
- **Security Scanning**: Scan images for vulnerabilities
- **Resource Limits**: Prevent resource exhaustion

### AWS Security

- **IAM Roles**: Minimal required permissions
- **Security Groups**: Restrictive inbound rules
- **VPC**: Isolated network environment
- **Encryption**: Data encrypted in transit and at rest

## 📈 Monitoring and Observability

### Metrics

- **System Metrics**: CPU, memory, disk usage
- **Application Metrics**: Processing rate, error rate, confidence scores
- **Business Metrics**: Active users, heart rate readings, face detection rate

### Logging

- **Structured Logging**: JSON format for easy parsing
- **Log Levels**: Configurable verbosity
- **Centralized Logs**: CloudWatch integration
- **Log Retention**: Configurable retention periods

### Alerting

- **Health Checks**: Service availability monitoring
- **Performance Alerts**: CPU/memory threshold alerts
- **Error Rate Alerts**: High error rate notifications
- **Custom Metrics**: Heart rate processing specific alerts

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run linting and tests
5. Submit a pull request

### Code Standards

- **Type Hints**: Use Python type hints
- **Documentation**: Document all public functions
- **Testing**: Add tests for new features
- **Linting**: Use black, flake8, and mypy

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Medical Disclaimer

This software is for **educational and demonstration purposes only**. It is not intended for medical diagnosis or monitoring. The heart rate measurements are estimates and should not be relied upon for medical decisions. Always consult healthcare professionals for medical advice.

---

**🎯 Ready to integrate heart rate detection into your application? Start with the Quick Start guide above!**