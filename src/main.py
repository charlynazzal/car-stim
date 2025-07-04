import cv2
import carla
from carla_connector import CarlaConnector
from lane_detection import LaneDetector
import sys

def main_loop(connector):
    """
    The main loop for processing and visualization with lane detection.
    """
    vehicle = connector.vehicle
    # Disable autopilot - we'll implement our own lane-following control
    vehicle.set_autopilot(False)
    
    # Keep vehicle stationary for now while we test lane detection
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))

    # Initialize lane detector
    lane_detector = LaneDetector()
    
    spectator = connector.world.get_spectator()

    try:
        while True:
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
            
            # Display results
            cv2.imshow("CARLA Camera - Lane Detection", result_image)
            
            # Optional: Show processing steps (press 'e' to toggle)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                cv2.imshow("Edges", lane_info['edges_image'])
                cv2.imshow("Masked Edges", lane_info['masked_edges'])
            elif key == ord('q'):
                break
                
            # Print enhanced lane detection info
            if lane_info['total_lines'] > 0:
                print(f"Total: {lane_info['total_lines']} | Left: {lane_info['left_lanes']} | Right: {lane_info['right_lanes']}")
                
    finally:
        cv2.destroyAllWindows()

def main():
    """
    Main function to initialize and run the simulation.
    """
    connector = None
    try:
        connector = CarlaConnector()
        connector.connect()
        
        vehicle = connector.spawn_vehicle()
        connector.setup_camera(attach_to=vehicle)

        main_loop(connector)

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    finally:
        if connector:
            connector.cleanup()

if __name__ == '__main__':
    main() 