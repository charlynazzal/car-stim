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
import carla
from carla_connector import CarlaConnector
from lane_detection import LaneDetector
from vehicle_controller import VehicleController
import sys

def main_loop(connector):
    """
    Enhanced main loop with robust lane following and obstacle avoidance.
    """
    vehicle = connector.vehicle
    
    # Initialize enhanced systems
    lane_detector = LaneDetector()
    vehicle_controller = VehicleController(vehicle)
    
    # Disable autopilot - we're implementing our own control
    vehicle.set_autopilot(False)
    
    spectator = connector.world.get_spectator()
    
    # Performance monitoring
    frame_count = 0
    successful_detections = 0

    try:
        print("Starting enhanced lane-following control...")
        print("Features: Road segmentation, obstacle detection, adaptive control")
        print("Press 'q' to quit")
        
        # Ensure vehicle is ready
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False))
        print("Vehicle ready - starting autonomous control...")
        
        while True:
            try:
                frame_count += 1
                
                # Position spectator to follow the vehicle
                transform = carla.Transform(
                    vehicle.get_transform().transform(carla.Location(x=-5, z=2.5)),
                    vehicle.get_transform().rotation
                )
                spectator.set_transform(transform)

                # Get camera image
                image = connector.image_queue.get()
                
                # Process image with enhanced lane detection
                result_image, lane_info = lane_detector.process_image(image)
                
                # Check for obstacles
                obstacles_detected = lane_info.get('obstacles_detected', 0) > 0
                
                # Apply enhanced vehicle control
                control_info = vehicle_controller.apply_control(
                    lane_info['lane_center'], 
                    obstacles_detected=obstacles_detected
                )
                
                # Track successful detections
                if lane_info['lane_center']['confidence'] in ['HIGH', 'MEDIUM']:
                    successful_detections += 1
                
                # Display results
                cv2.imshow("CARLA Camera - Enhanced Lane Following", result_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset controller
                    vehicle_controller.reset_controller()
                    print("Controller reset")
                elif key == ord('s'):
                    # Show statistics
                    stats = vehicle_controller.get_performance_stats()
                    detection_rate = (successful_detections / frame_count) * 100 if frame_count > 0 else 0
                    print(f"\n=== Performance Statistics ===")
                    print(f"Detection Success Rate: {detection_rate:.1f}%")
                    print(f"Average Steering Error: {stats['average_steering_error']:.1f}px")
                    print(f"Lane Following Active: {stats['lane_following_active']}")
                    print(f"Frames Controlled: {stats['frames_controlled']}")
                    print("=" * 30)
                    
                # Enhanced console output
                if lane_info and control_info:
                    lane_center = lane_info['lane_center']
                    left_pos = f"{lane_center['left_lane_x']:.1f}" if lane_center['left_lane_x'] is not None else "None"
                    right_pos = f"{lane_center['right_lane_x']:.1f}" if lane_center['right_lane_x'] is not None else "None"
                    center_pos = f"{lane_center['lane_center_x']:.1f}" if lane_center['lane_center_x'] is not None else "None"
                    error = f"{lane_center['steering_error']:.1f}px" if lane_center['steering_error'] is not None else "None"
                    
                    # Enhanced status display
                    status_indicators = []
                    if lane_info.get('road_area_detected', False):
                        status_indicators.append("ROAD")
                    if obstacles_detected:
                        status_indicators.append("OBS")
                    if control_info.get('lane_following_active', False):
                        status_indicators.append("ACTIVE")
                    
                    status_str = "|".join(status_indicators) if status_indicators else "SEARCHING"
                    
                    print(f"L:{left_pos} R:{right_pos} C:{center_pos} E:{error} "
                          f"Conf:{lane_center['confidence']} Steer:{control_info['steering_angle']:.3f} "
                          f"Throttle:{control_info['throttle']:.2f} Status:{status_str}")
                    
                    # Warning for critical situations
                    if control_info['safety_status'] == 'EMERGENCY_BRAKE':
                        print("⚠️  EMERGENCY BRAKE ACTIVATED!")
                    elif obstacles_detected:
                        print("⚠️  OBSTACLES DETECTED - SLOWING DOWN")
                    elif lane_center['confidence'] == 'NONE':
                        print("⚠️  NO LANE DETECTION - DEGRADED MODE")
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                # Apply safe stop in case of error
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5))
                break
                
    except KeyboardInterrupt:
        print("\nStopping enhanced lane-following control...")
    finally:
        # Stop the vehicle safely
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        cv2.destroyAllWindows()
        
        # Print final statistics
        if frame_count > 0:
            detection_rate = (successful_detections / frame_count) * 100
            print(f"\nFinal Performance:")
            print(f"- Total frames processed: {frame_count}")
            print(f"- Successful detections: {detection_rate:.1f}%")
            stats = vehicle_controller.get_performance_stats()
            print(f"- Average steering error: {stats['average_steering_error']:.1f}px")
            print(f"- Lane following was active: {stats['lane_following_active']}")

def main():
    """
    Main function to initialize and run the enhanced lane-following simulation.
    """
    connector = None
    try:
        print("Initializing Enhanced CARLA Lane-Following System...")
        print("Features: Road segmentation, obstacle detection, adaptive control")
        connector = CarlaConnector()
        connector.connect()
        
        vehicle = connector.spawn_vehicle()
        connector.setup_camera(attach_to=vehicle)
        
        print("Enhanced system initialized successfully!")
        print("\nControls:")
        print("  q - Quit")
        print("  r - Reset controller")
        print("  s - Show statistics")
        print()
        main_loop(connector)

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    finally:
        if connector:
            connector.cleanup()
            print("Cleanup completed.")

if __name__ == '__main__':
    main() 