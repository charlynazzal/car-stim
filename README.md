# Modern Autonomous Driving System

An autonomous driving system using YOLOv8 for object detection and semantic segmentation for lane detection in the CARLA simulator.

## Features

- YOLOv8 object detection for vehicles, pedestrians, and traffic signs
- Semantic segmentation for precise lane detection
- Intelligent safety assessment and emergency response
- Adaptive vehicle control with object awareness
- GPU acceleration support
- Real-time performance monitoring

## Prerequisites

- Python 3.8+
- CARLA Simulator 0.9.13+
- NVIDIA GPU with CUDA support (recommended)

## Installation

### Quick Setup
```bash
python setup_modern_cv.py
```

### Manual Setup
```bash
pip install -r requirements.txt
mkdir models data logs outputs
```

## Usage

1. Start CARLA simulator
2. Run the system:
   ```bash
   python src/main.py
   ```

### Controls
- 'q' or ESC: Quit
- 'r': Reset controllers
- 's': Show statistics
- 'p': Performance metrics
- 'h': Help

## System Architecture

The system consists of three main components:

1. **ModernLaneDetector**: YOLOv8 object detection and UNet semantic segmentation
2. **ModernVehicleController**: Adaptive PID control with safety features
3. **CarlaConnector**: Enhanced CARLA interface with robust error handling

## Configuration

### Model Selection
- yolov8n: Fastest, lowest accuracy
- yolov8s: Balanced performance
- yolov8m: Better accuracy
- yolov8l: High accuracy
- yolov8x: Best accuracy

### Performance Tuning
```python
# Adjust in lane_detection.py
lane_detector.yolo_model = YOLO('yolov8s.pt')

# Adjust in vehicle_controller.py
vehicle_controller.kp_steer = 0.003
vehicle_controller.base_speed = 35.0
```

## Technical Details

### Object Detection
- Vehicles: cars, motorcycles, buses, trucks
- Pedestrians: people detection
- Traffic signs: stop signs, traffic lights

### Lane Segmentation
- Ego lane: current driving lane
- Adjacent lanes: neighboring lanes
- Road boundaries: road edges
- Drivable area: safe driving regions

### Safety Features
- Emergency braking for critical situations
- Adaptive speed control based on safety assessment
- Object avoidance steering adjustments
- Continuous lane center tracking

## Troubleshooting

**CARLA Connection Issues**:
- Ensure CARLA simulator is running
- Check host/port configuration

**Performance Issues**:
- Use smaller YOLO model (yolov8n)
- Enable GPU acceleration
- Reduce image resolution

**Detection Issues**:
- Verify model files are downloaded
- Check lighting conditions in CARLA
- Adjust confidence thresholds

## Development

The system is modular and extensible. Key files:

- `src/lane_detection.py`: Computer vision pipeline
- `src/vehicle_controller.py`: Vehicle control logic
- `src/carla_connector.py`: CARLA interface
- `src/main.py`: Main application loop

## Status

This is a work in progress implementing modern computer vision techniques for autonomous driving research and development.
