# Crowd Safety Project - Deployment Guide

## Dependencies Installation

### 1. Python Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. System Dependencies

Make sure you have Python 3.8+ installed on your system.

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ankitsharma003/crowd_safety_project.git
cd crowd_safety_project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## Production Deployment

### Option 1: Using Gunicorn (Recommended)

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Option 2: Using Waitress (Windows-friendly)

```bash
# Install waitress
pip install waitress

# Run with waitress
waitress-serve --host=0.0.0.0 --port=8000 app:app
```

### Option 3: Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:

```bash
docker build -t crowd-safety .
docker run -p 5000:5000 crowd-safety
```

## Required Files for Deployment

Make sure these files are present:

- `app.py` - Main Flask application
- `flow_estimation.py` - Flow detection algorithms
- `movement_instruction.py` - Movement guidance system
- `bottleneck_detection.py` - Bottleneck detection
- `venue_config.py` - Venue configuration
- `autoencoder_weights.pkl` - Pre-trained model weights
- `pca_results.pkl` - PCA transformation data
- `spatiotemporal_matrix.pkl` - Spatiotemporal data
- `anomalies.pkl` - Anomaly detection data

## Environment Variables (Optional)

You can set these environment variables for configuration:

```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
```

## Troubleshooting

### Common Issues:

1. **Missing .pkl files**: Ensure all model files are present
2. **Memory issues**: The PCA and model files are large (~14MB total)
3. **Port conflicts**: Change the port if 5000 is occupied
4. **Permission errors**: Run with appropriate permissions

### Performance Optimization:

1. **Use a reverse proxy** (nginx) for production
2. **Enable gzip compression**
3. **Use a CDN** for static assets
4. **Monitor memory usage** due to large model files

## Cloud Deployment Options

### Heroku

1. Create `Procfile`:

```
web: gunicorn -w 4 -b 0.0.0.0:$PORT app:app
```

2. Deploy with Heroku CLI

### AWS/GCP/Azure

- Use containerized deployment
- Ensure sufficient memory allocation (512MB+)
- Use load balancers for high availability

## Security Considerations

1. **Never commit sensitive data** (API keys, passwords)
2. **Use environment variables** for configuration
3. **Enable HTTPS** in production
4. **Implement rate limiting** for API endpoints
5. **Validate file uploads** (image types, sizes)
