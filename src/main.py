"""
Enhanced Autonomous Driving Lane Detection System

This system demonstrates robust lane detection and vehicle control using CARLA simulator.

KEY IMPROVEMENTS:
1. Road Segmentation: Identifies drivable areas before lane detection
2. Robust Lane Detection: Only detects lanes within road boundaries
3. Obstacle Detection: Basic obstacle awareness and avoidance
4. Enhanced Control: Improved PID control with safety mechanisms
5. Temporal Filtering: Stable detection across multiple frames

Features:
- Road area segmentation for focused lane detection
- Multi-method lane line detection (color + texture + gradient)
- Basic obstacle detection and avoidance
- Adaptive speed control based on road conditions
- Emergency braking for safety-critical situations
- Enhanced temporal smoothing for stable control
"""

import cv2
import numpy as np
import carla
import time
import random
import logging
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.carla_connector import CarlaConnector
from src.lane_detection import ModernLaneDetector
from src.vehicle_controller import ModernVehicleController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def spawn_vehicle(world):
    """Spawn a vehicle in the world."""
    try:
        blueprint_library = world.get_blueprint_library()
        
        # Try different vehicle types if Tesla Model 3 is not available
        vehicle_blueprints = [
            'vehicle.tesla.model3',
            'vehicle.audi.a2',
            'vehicle.bmw.grandtourer',
            'vehicle.chevrolet.impala',
            'vehicle.dodge.charger_police',
            'vehicle.ford.mustang',
            'vehicle.lincoln.mkz_2017',
            'vehicle.mercedes.coupe',
            'vehicle.mini.cooper_s',
            'vehicle.nissan.patrol',
            'vehicle.toyota.prius'
        ]
        
        vehicle_bp = None
        for bp_name in vehicle_blueprints:
            try:
                filtered_bps = blueprint_library.filter(bp_name)
                if filtered_bps:
                    vehicle_bp = filtered_bps[0]
                    logger.info(f"Using vehicle blueprint: {bp_name}")
                    break
            except Exception as e:
                logger.warning(f"Blueprint {bp_name} not available: {e}")
                continue
        
        if not vehicle_bp:
            # Fallback to any available vehicle
            all_vehicles = blueprint_library.filter('vehicle.*')
            if all_vehicles:
                vehicle_bp = all_vehicles[0]
                logger.info(f"Using fallback vehicle blueprint: {vehicle_bp.id}")
            else:
                logger.error("No vehicle blueprints available")
                return None
        
        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()
        
        if not spawn_points:
            logger.error("No spawn points available on this map")
            return None
        
        logger.info(f"Found {len(spawn_points)} spawn points")
        
        # Try multiple spawn points if needed
        for attempt in range(min(10, len(spawn_points))):
            try:
                spawn_point = spawn_points[attempt]
                
                # Check if spawn point is clear
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                if vehicle:
                    logger.info(f"Vehicle spawned at spawn point {attempt}: {spawn_point.location}")
                    return vehicle
                else:
                    logger.warning(f"Spawn point {attempt} is occupied, trying next...")
                    
            except Exception as e:
                logger.warning(f"Failed to spawn at point {attempt}: {e}")
                continue
        
        logger.error("Failed to spawn vehicle at any available spawn point")
        return None
        
    except Exception as e:
        logger.error(f"Error spawning vehicle: {e}")
        return None

def main_loop(connector, headless=False):
    """Main control loop for the modern autonomous driving system."""
    print("=" * 60)
    print("MODERN AUTONOMOUS DRIVING SYSTEM")
    print("   Using YOLOv8 + Semantic Segmentation")
    if headless:
        print("   Running in HEADLESS mode (no GUI)")
    print("=" * 60)
    
    try:
        # Initialize components
        print("Initializing modern computer vision system...")
        
        # Initialize lane detector with modern CV
        lane_detector = ModernLaneDetector()
        logger.info("Modern lane detector initialized")
        
        # Spawn vehicle
        vehicle = spawn_vehicle(connector.world)
        if not vehicle:
            logger.error("Failed to spawn vehicle")
            return
        
        # Initialize vehicle controller
        vehicle_controller = ModernVehicleController(vehicle)
        logger.info("Modern vehicle controller initialized")
        
        # Setup camera
        camera = connector.setup_camera(vehicle)
        if not camera:
            logger.error("Failed to setup camera")
            return
        
        # System status
        detector_status = lane_detector.get_status()
        controller_status = vehicle_controller.get_status()
        
        print(f"System Status:")
        print(f"   YOLOv8 Available: {detector_status['yolo_available']}")
        print(f"   Segmentation Available: {detector_status['segmentation_available']}")
        print(f"   Device: {detector_status['device']}")
        print(f"   Controller Active: {controller_status['is_active']}")
        print()
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        performance_log_interval = 30  # seconds
        last_performance_log = time.time()
        
        # Statistics
        detection_stats = {
            'total_frames': 0,
            'successful_detections': 0,
            'object_detections': 0,
            'safety_warnings': 0,
            'emergency_brakes': 0
        }
        
        print("Starting autonomous driving loop...")
        if not headless:
            print("Controls:")
            print("  'q' or ESC: Quit")
            print("  'r': Reset controllers")
            print("  's': Show statistics")
            print("  'p': Performance metrics")
            print("  'h': Help")
        else:
            print("Running in headless mode. Press Ctrl+C to stop.")
        print()
        
        while True:
            frame_start = time.time()
            
            # Get camera image
            image = connector.get_camera_image()
            if image is None:
                continue
            
            # Process frame with modern computer vision
            cv_results = lane_detector.process_frame(image)
            
            # Apply vehicle control
            control_results = vehicle_controller.apply_control(cv_results)
            
            # Update statistics
            detection_stats['total_frames'] += 1
            if cv_results.get('lane_confidence', 0) > 0.5:
                detection_stats['successful_detections'] += 1
            
            total_objects = sum(len(objects) for objects in cv_results.get('objects', {}).values())
            if total_objects > 0:
                detection_stats['object_detections'] += 1
            
            safety = cv_results.get('safety', {})
            if safety.get('warnings'):
                detection_stats['safety_warnings'] += 1
            if safety.get('emergency_brake', False):
                detection_stats['emergency_brakes'] += 1
            
            # Visualize results (only if not headless)
            if not headless:
                try:
                    vis_image = lane_detector.visualize_results(image, cv_results)
                    
                    # Add control information to visualization
                    add_control_info(vis_image, control_results, detection_stats)
                    
                    # Display image
                    cv2.imshow('Modern Autonomous Driving', vis_image)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        print("Stopping autonomous driving system...")
                        break
                    elif key == ord('r'):
                        print("Resetting controllers...")
                        vehicle_controller.reset_controller()
                        lane_detector = ModernLaneDetector()  # Reinitialize detector
                        print("Controllers reset")
                    elif key == ord('s'):
                        show_statistics(detection_stats, time.time() - start_time)
                    elif key == ord('p'):
                        show_performance_metrics(lane_detector, vehicle_controller)
                    elif key == ord('h'):
                        show_help()
                except Exception as gui_error:
                    logger.warning(f"GUI error (switching to headless): {gui_error}")
                    headless = True
                    print("Switched to headless mode due to GUI error")
            else:
                # In headless mode, just log progress occasionally
                if detection_stats['total_frames'] % 30 == 0:
                    success_rate = detection_stats['successful_detections'] / max(detection_stats['total_frames'], 1) * 100
                    print(f"Frame {detection_stats['total_frames']}: Success rate {success_rate:.1f}%, "
                          f"Objects: {detection_stats['object_detections']}, "
                          f"Warnings: {detection_stats['safety_warnings']}")
            
            # Performance logging
            current_time = time.time()
            if current_time - last_performance_log > performance_log_interval:
                log_performance(detection_stats, current_time - start_time, lane_detector, vehicle_controller)
                last_performance_log = current_time
            
            frame_count += 1
            
            # Frame rate control
            frame_time = time.time() - frame_start
            if frame_time < 0.033:  # Target ~30 FPS
                time.sleep(0.033 - frame_time)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        print(f"Error: {e}")
    finally:
        # Cleanup
        print("Cleaning up...")
        try:
            if 'vehicle' in locals() and vehicle:
                vehicle.destroy()
                logger.info("Vehicle destroyed")
        except:
            pass
        
        if not headless:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        connector.cleanup()
        print("Cleanup complete")

def add_control_info(image, control_results, stats):
    """Add control information to the visualization."""
    if 'error' in control_results:
        cv2.putText(image, f"CONTROL ERROR: {control_results['error']}", 
                   (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return
    
    # Control values
    y_pos = image.shape[0] - 140
    cv2.putText(image, f"Steering: {control_results.get('steering', 0):.3f}", 
               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(image, f"Throttle: {control_results.get('throttle', 0):.3f}", 
               (10, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(image, f"Brake: {control_results.get('brake', 0):.3f}", 
               (10, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(image, f"Speed: {control_results.get('target_speed', 0):.1f} km/h", 
               (10, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Safety override indicator
    if control_results.get('safety_override', False):
        cv2.putText(image, "EMERGENCY BRAKE", 
                   (image.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Frame counter
    cv2.putText(image, f"Frame: {stats['total_frames']}", 
               (image.shape[1] - 150, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def show_statistics(stats, elapsed_time):
    """Display system statistics."""
    print("\n📈 SYSTEM STATISTICS")
    print("=" * 40)
    print(f"Runtime: {elapsed_time:.1f} seconds")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Successful detections: {stats['successful_detections']} ({stats['successful_detections']/max(stats['total_frames'], 1)*100:.1f}%)")
    print(f"Object detections: {stats['object_detections']} ({stats['object_detections']/max(stats['total_frames'], 1)*100:.1f}%)")
    print(f"Safety warnings: {stats['safety_warnings']}")
    print(f"Emergency brakes: {stats['emergency_brakes']}")
    print(f"Average FPS: {stats['total_frames']/elapsed_time:.1f}")
    print("=" * 40)

def show_performance_metrics(lane_detector, vehicle_controller):
    """Display performance metrics."""
    print("\n⚡ PERFORMANCE METRICS")
    print("=" * 40)
    
    # Lane detector performance
    detector_status = lane_detector.get_status()
    print(f"Lane Detection:")
    print(f"  • Average FPS: {detector_status.get('avg_fps', 0):.1f}")
    print(f"  • Device: {detector_status.get('device', 'Unknown')}")
    print(f"  • Models loaded: {detector_status.get('models_loaded', False)}")
    
    # Vehicle controller performance
    controller_metrics = vehicle_controller.get_performance_metrics()
    if controller_metrics:
        print(f"Vehicle Control:")
        print(f"  • Avg control time: {controller_metrics.get('avg_control_time', 0)*1000:.1f}ms")
        print(f"  • Avg steering: {controller_metrics.get('avg_steering', 0):.3f}")
        print(f"  • Avg lane confidence: {controller_metrics.get('avg_lane_confidence', 0):.3f}")
        print(f"  • Avg safety score: {controller_metrics.get('avg_safety_score', 0):.3f}")
    
    print("=" * 40)

def show_help():
    """Display help information."""
    print("\n❓ HELP - MODERN AUTONOMOUS DRIVING SYSTEM")
    print("=" * 50)
    print("This system uses:")
    print("  • YOLOv8 for real-time object detection")
    print("  • Semantic segmentation for lane detection")
    print("  • Advanced safety assessment")
    print("  • Intelligent vehicle control")
    print()
    print("Controls:")
    print("  • 'q' or ESC: Quit system")
    print("  • 'r': Reset controllers")
    print("  • 's': Show statistics")
    print("  • 'p': Performance metrics")
    print("  • 'h': This help")
    print()
    print("Visual Indicators:")
    print("  • Red boxes: Vehicles")
    print("  • Orange boxes: Pedestrians")
    print("  • Green boxes: Traffic signs")
    print("  • White overlay: Ego lane")
    print("  • Yellow overlay: Adjacent lanes")
    print("  • Blue overlay: Drivable area")
    print("  • Yellow line: Lane center")
    print("=" * 50)

def log_performance(stats, elapsed_time, lane_detector, vehicle_controller):
    """Log performance metrics."""
    fps = stats['total_frames'] / elapsed_time
    success_rate = stats['successful_detections'] / max(stats['total_frames'], 1) * 100
    
    logger.info(f"Performance - FPS: {fps:.1f}, Success Rate: {success_rate:.1f}%, "
                f"Objects: {stats['object_detections']}, Warnings: {stats['safety_warnings']}")

def main():
    """Main function."""
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            os.makedirs('models')
            print("Created models directory for pretrained weights")
        
        # Initialize CARLA connector
        connector = CarlaConnector()
        
        # Check if GUI is available (test OpenCV display)
        headless = False
        try:
            # Try to create a small test window
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('GUI Test', test_img)
            cv2.waitKey(1)
            cv2.destroyWindow('GUI Test')
            print("GUI mode available")
        except Exception as gui_error:
            print(f"GUI not available, running in headless mode: {gui_error}")
            headless = True
        
        # Start main loop
        main_loop(connector, headless=headless)
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        print(f"Failed to start: {e}")
        print("\nTroubleshooting:")
        print("  Make sure CARLA simulator is running")
        print("  Check that all dependencies are installed")
        print("  Run: pip install -r requirements.txt")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 