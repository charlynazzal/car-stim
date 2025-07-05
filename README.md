# CARLA Lane Following System

A Python-based autonomous lane following system using the CARLA simulator.

## Features

- **CARLA Connection Module**: Connects to CARLA simulator and manages vehicle/sensor spawning
- **Lane Detection Module**: Real-time lane detection using OpenCV (Canny edge detection, Hough transforms)
- **Vehicle Control Module**: PID-based steering control with safety limits

## Prerequisites

1. **CARLA Simulator**: Download from [CARLA Releases](https://github.com/carla-simulator/carla/releases/)
   - Tested with CARLA 0.10.0
   - Extract to any location on your system

2. **Python 3.10**: Required for CARLA client compatibility

## Setup

1. **Clone the repository**:
   ```bash[
   git clone https://github.com/charlynazzal/car-stim
   cd autonomous-driving
   ```

2. **Create virtual environment**:
   ```bash
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. **Install CARLA client**:
   ```bash
   # Navigate to your CARLA installation
   pip install "path\to\CARLA\PythonAPI\carla\dist\carla-0.10.0-cp310-cp310-win_amd64.whl"
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start CARLA Simulator**:
   - Run `CarlaUnreal.exe` from your CARLA installation
   - Or use: `Start-Process "path\to\CarlaUnreal.exe"`

2. **Run the lane following system**:
   ```bash
   python src/main.py
   ```

3. **Controls**:
   - Press `q` to quit the camera view
   - The vehicle will drive on autopilot initially

## Project Structure

```
src/
├── carla_connector.py    # CARLA connection and sensor management
├── main.py              # Main application entry point
├── lane_detection.py    # Lane detection module (coming soon)
└── vehicle_control.py   # PID controller (coming soon)
```

## Development

The system is designed to work independently of where CARLA is installed. Your Python scripts connect to the CARLA server via TCP (localhost:2000), so you can:

- Run CARLA from anywhere on your system
- Develop and test your Python code separately
- Make changes to the scripts and see results immediately

## Next Steps

- [ ] Implement lane detection using OpenCV
- [ ] Add PID controller for steering
- [ ] Include safety limits and error handling
- [ ] Add visualization overlays 
