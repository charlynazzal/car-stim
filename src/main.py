"""
Autonomous Driving Lane Detection System - OPTIMIZED VERSION

This system demonstrates lane detection and vehicle control using CARLA simulator.

MAJOR OPTIMIZATIONS IMPLEMENTED:
1. Camera Performance:
   - Reduced resolution to 480x360 for better performance
   - Lowered FPS to 20 for reduced lag
   - Disabled post-processing effects
   - Optimized sensor tick rate

2. Noise Reduction:
   - Enhanced Gaussian blur (7x7 kernel)
   - Morphological operations for noise removal
   - Stronger outlier filtering
   - Position change limits (30px max per frame)

3. Steering Stability:
   - Reduced PID gains for smoother control
   - Enhanced hysteresis with confidence-based dead zones
   - Steering rate limiting (0.1 max change per frame)
   - Improved temporal smoothing (15-frame history)

4. Lane Boundary Protection:
   - Boundary validation to prevent illegal lane changes
   - Lane width validation (60-200px range)
   - Conservative lane center estimation (100px width)
   - Image boundary constraints (25%-75% of width)

5. Enhanced Temporal Filtering:
   - Exponential weighted averaging
   - Confidence-based weight adjustment
   - 5-frame minimum for stable decisions
   - 25px max lane center change per frame

Expected Results:
- Smoother, less jerky steering
- Reduced camera lag and better performance
- Prevention of illegal lane changes
- More stable lane following behavior
- Reduced noise and oscillation
"""

import cv2
import carla
from carla_connector import CarlaConnector
from lane_detection import LaneDetector
from vehicle_controller import VehicleController
import sys

def main_loop(connector):
    """
    The main loop for lane-following with vehicle control.
    """
    vehicle = connector.vehicle
    
    # Initialize lane detector and vehicle controller
    lane_detector = LaneDetector()
    vehicle_controller = VehicleController(vehicle)
    
    # Disable autopilot - we're implementing our own control
    vehicle.set_autopilot(False)
    
    spectator = connector.world.get_spectator()

    try:
        print("Starting lane-following control...")
        print("Press 'q' to quit")
        
        # Ensure vehicle is ready to move - release any brakes
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False))
        print("Vehicle brakes released, ready to start lane following...")
        
        while True:
            try:
                # Position spectator to follow the vehicle
                transform = carla.Transform(
                    vehicle.get_transform().transform(carla.Location(x=-5, z=2.5)),
                    vehicle.get_transform().rotation
                )
                spectator.set_transform(transform)

                # Get camera image
                image = connector.image_queue.get()
                
                # Process image with lane detection
                result_image, lane_info = lane_detector.process_image(image)
                
                # Apply vehicle control based on lane detection
                control_info = vehicle_controller.apply_control(lane_info['lane_center'])
                
                # Display results
                cv2.imshow("CARLA Camera - Lane Following", result_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
                # Enhanced console output with control information
                if lane_info and control_info:
                    lane_center = lane_info['lane_center']
                    left_pos = f"{lane_center['left_lane_x']:.1f}" if lane_center['left_lane_x'] is not None else "None"
                    right_pos = f"{lane_center['right_lane_x']:.1f}" if lane_center['right_lane_x'] is not None else "None"
                    center_pos = f"{lane_center['lane_center_x']:.1f}" if lane_center['lane_center_x'] is not None else "None"
                    error = f"{lane_center['steering_error']:.1f}px" if lane_center['steering_error'] is not None else "None"
                    
                    print(f"Left: {left_pos} | Right: {right_pos} | Center: {center_pos} | "
                          f"Error: {error} | Conf: {lane_center['confidence']} | "
                          f"Steer: {control_info['steering_angle']:.3f} | "
                          f"Throttle: {control_info['throttle']:.2f} | Status: {control_info['safety_status']}")
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                break
    except KeyboardInterrupt:
        print("\nStopping lane-following control...")
    finally:
        # Stop the vehicle safely
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        cv2.destroyAllWindows()

def main():
    """
    Main function to initialize and run the lane-following simulation.
    """
    connector = None
    try:
        print("Initializing CARLA Lane-Following System...")
        connector = CarlaConnector()
        connector.connect()
        
        vehicle = connector.spawn_vehicle()
        connector.setup_camera(attach_to=vehicle)
        
        print("System initialized successfully!")
        main_loop(connector)

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    finally:
        if connector:
            connector.cleanup()
            print("Cleanup completed.")

if __name__ == '__main__':
    main() 