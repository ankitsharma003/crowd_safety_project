# 🚶‍♂️ Crowd Safety & Flow Analysis System

A real-time crowd flow analysis and safety guidance system that detects movement patterns, identifies bottlenecks, and provides intelligent evacuation guidance using computer vision and machine learning.

## 🌟 Features

- **Real-time Flow Detection**: Advanced optical flow algorithms to detect crowd movement patterns
- **Bottleneck Detection**: Identifies congested areas and potential safety hazards
- **Intelligent Guidance**: Provides directional recommendations for optimal crowd flow
- **Interactive Web Interface**: User-friendly Flask web application with real-time visualization
- **Anomaly Detection**: Machine learning-based anomaly detection using autoencoders
- **Multi-Exit Venue Support**: Configurable venue layouts with multiple exit points
- **Flow Visualization**: Real-time arrow overlays showing movement direction and intensity

## 🎯 Use Cases

- **Event Management**: Stadiums, concerts, festivals
- **Emergency Evacuation**: Shopping malls, airports, train stations
- **Crowd Control**: Public spaces, transportation hubs
- **Safety Monitoring**: High-density pedestrian areas
- **Traffic Management**: Pedestrian flow optimization

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (for model files)
- Web browser

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ankitsharma003/crowd_safety_project.git
cd crowd_safety_project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Open your browser**
Navigate to `http://localhost:5000`

## 📊 How It Works

### 1. Flow Detection Pipeline
```
Input Images → Frame Differencing → Gradient Analysis → Flow Estimation → Direction Analysis
```

### 2. Movement Analysis
- **Phase Correlation**: Detects global shifts between frames
- **Gradient Analysis**: Identifies local movement patterns
- **Confidence Scoring**: Validates detection reliability
- **Fallback Mechanisms**: Ensures robust operation

### 3. Safety Guidance
- **Zone Detection**: Identifies current location in venue
- **Exit Optimization**: Finds nearest safe exit
- **Bottleneck Avoidance**: Suggests alternative routes
- **Real-time Updates**: Continuous guidance updates

## 🖥️ Web Interface

### Upload Images
- Upload multiple consecutive frames
- Supports common image formats (PNG, JPG, TIFF)
- Real-time processing and analysis

### Visualization
- **Flow Arrows**: Red arrows showing movement direction
- **Confidence Indicators**: Color-coded confidence levels
- **Zone Mapping**: Current location and exit information
- **Bottleneck Alerts**: Warning indicators for congested areas

### Guidance Cards
- **Frame-by-frame Analysis**: Detailed information for each frame
- **Movement Recommendations**: Directional guidance
- **Exit Instructions**: Step-by-step evacuation guidance
- **Confidence Scores**: Reliability indicators

## 🏗️ Project Structure

```
crowd_safety_project/
├── app.py                          # Main Flask web application
├── flow_estimation.py              # Flow detection algorithms
├── movement_instruction.py         # Guidance generation
├── bottleneck_detection.py         # Congestion detection
├── venue_config.py                # Venue layout configuration
├── autoencoder.py                 # ML model implementation
├── anomaly_detection.py           # Anomaly detection
├── visualization.py               # Visualization utilities
├── guide_cli.py                  # Command-line interface
├── run_pipeline.py               # Pipeline execution
├── requirements.txt              # Python dependencies
├── DEPLOYMENT.md                 # Deployment guide
└── README.md                     # This file
```

## 🔧 Configuration

### Venue Setup
Edit `venue_config.py` to configure your venue:
- Exit locations
- Zone definitions
- Safety parameters
- Layout specifications

### Detection Parameters
Adjust sensitivity in the web interface:
- **Density Threshold**: Crowd density sensitivity
- **Movement Threshold**: Motion detection sensitivity
- **Alpha Parameter**: Flow smoothing factor

## 📈 Performance

### System Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 4GB+ (model files are ~14MB)
- **Storage**: 100MB+ for application and models
- **Network**: Stable internet for web interface

### Optimization Tips
- Use smaller image sizes for faster processing
- Adjust detection thresholds based on venue
- Monitor memory usage with large datasets
- Use production WSGI server for deployment

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Structure
- **Modular Design**: Separate concerns for flow, guidance, and visualization
- **Error Handling**: Robust error handling and fallback mechanisms
- **Performance**: Optimized algorithms for real-time processing
- **Extensibility**: Easy to add new detection methods

### Adding New Features
1. **New Detection Methods**: Add to `flow_estimation.py`
2. **Guidance Algorithms**: Extend `movement_instruction.py`
3. **Visualization**: Modify `visualization.py`
4. **Web Interface**: Update `app.py` templates

## 🚀 Deployment

### Production Deployment
See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

### Quick Production Setup
```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Docker Deployment
```bash
# Build image
docker build -t crowd-safety .

# Run container
docker run -p 5000:5000 crowd-safety
```

## 📊 Model Files

The system uses pre-trained models:
- `autoencoder_weights.pkl` - Neural network weights
- `pca_results.pkl` - Dimensionality reduction
- `spatiotemporal_matrix.pkl` - Spatiotemporal patterns
- `anomalies.pkl` - Anomaly detection data

## 🔒 Security Considerations

- **File Upload Validation**: Image type and size validation
- **Input Sanitization**: Secure handling of user inputs
- **Rate Limiting**: Prevent abuse of the system
- **HTTPS**: Use secure connections in production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/ankitsharma003/crowd_safety_project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ankitsharma003/crowd_safety_project/discussions)

## 🙏 Acknowledgments

- **UCSD Anomaly Dataset**: For providing test data
- **OpenCV Community**: For computer vision algorithms
- **Flask Community**: For web framework
- **NumPy/SciPy**: For scientific computing

## 📚 References

- Crowd flow analysis algorithms
- Computer vision techniques
- Machine learning for anomaly detection
- Real-time web applications
- Safety engineering principles

---

**⚠️ Important**: This system is designed for research and educational purposes. For production safety-critical applications, ensure proper testing and validation.
