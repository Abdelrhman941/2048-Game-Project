# 2048 DQN Flask App - Docker Deployment

This guide shows how to containerize and deploy the 2048 Deep Q-Network Flask application using Docker.

## 🐳 Quick Start [open docker desktop first]

### 1. Build the Docker Image
```bash
docker build -t 2048-dqn-flask .
```

### 2. Run the Container
```bash
docker run -p 5000:5000 2048-dqn-flask
```

### 3. Access the Application
Open your browser and go to:
```
http://localhost:5000
```

## 🚀 Alternative: Using Docker Compose

### Build and Run
```bash
docker-compose up --build
```

### Run in Background
```bash
docker-compose up -d
```

### Stop the Application
```bash
docker-compose down
```

## 📁 Project Structure for Docker

```
2048-DeepRL/
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore           # Files to exclude from Docker build
├── requirements.txt        # Python dependencies
├── flask-app/              # Flask application
│   ├── app.py              # Main Flask server
│   ├── templates/          # HTML templates
│   └── static/             # CSS, JS, assets
├── Model/                  # Trained models (mounted as volume)
│   └── run-*/
│       └── *.keras
├── game2048/               # Core game logic
└── README.md
```

## 🔧 Docker Configuration Details

### Dockerfile Features
- **Base Image**: Python 3.11 slim for optimized size
- **Dependencies**: Installs system packages and Python requirements
- **Security**: Non-root user execution
- **Health Check**: Automatic container health monitoring
- **Port**: Exposes port 5000 for Flask application

### Environment Variables
- `FLASK_APP=flask-app/app.py`
- `FLASK_ENV=production`
- `PYTHONDONTWRITEBYTECODE=1`
- `PYTHONUNBUFFERED=1`

## 📦 Deployment Options

### Local Development
```bash
# Build image
docker build -t 2048-dqn-flask .

# Run with volume mounting for live model updates
docker run -p 5000:5000 -v $(pwd)/Model:/app/Model 2048-dqn-flask
```

### Production Deployment
```bash
# Build optimized image
docker build -t 2048-dqn-flask:latest .

# Run with restart policy
docker run -d -p 5000:5000 --restart unless-stopped 2048-dqn-flask:latest
```

### Cloud Deployment (AWS/GCP/Azure)
```bash
# Tag for registry
docker tag 2048-dqn-flask:latest your-registry/2048-dqn-flask:latest

# Push to registry
docker push your-registry/2048-dqn-flask:latest

# Deploy on cloud platform
# (platform-specific commands)
```

## 🛠️ Customization

### Custom Port
```bash
docker run -p 8080:5000 2048-dqn-flask
```

### Environment Override
```bash
docker run -p 5000:5000 -e FLASK_DEBUG=1 2048-dqn-flask
```

### Volume Mounting
```bash
# Mount model directory
docker run -p 5000:5000 -v /path/to/models:/app/Model 2048-dqn-flask

# Mount entire project for development
docker run -p 5000:5000 -v $(pwd):/app 2048-dqn-flask
```

## 🔍 Troubleshooting

### View Container Logs
```bash
docker logs <container-id>
```

### Access Container Shell
```bash
docker exec -it <container-id> /bin/bash
```

### Check Container Health
```bash
docker ps
# Look for health status in output
```

### Common Issues

1. **Port Already in Use**
   ```bash
   # Use different port
   docker run -p 8080:5000 2048-dqn-flask
   ```

2. **Model Files Not Found**
   ```bash
   # Ensure Model directory exists and has correct permissions
   mkdir -p Model
   docker run -v $(pwd)/Model:/app/Model -p 5000:5000 2048-dqn-flask
   ```

3. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   docker run --memory=4g -p 5000:5000 2048-dqn-flask
   ```

## 📋 Requirements

- Docker 20.10+
- Docker Compose 2.0+ (optional)
- 2GB+ RAM for TensorFlow
- Trained model files in `Model/` directory

## 🌐 Sharing Your Application

To share with others:

1. **Share the Project**
   ```bash
   # Clone or download the project
   git clone <your-repo>
   cd 2048-DeepRL
   ```

2. **Build and Run**
   ```bash
   docker build -t 2048-dqn-flask .
   docker run -p 5000:5000 2048-dqn-flask
   ```

3. **Access the Game**
   ```
   http://localhost:5000
   ```

That's it! The complete 2048 DQN experience in a container! 🎮