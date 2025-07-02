import cv2
import carla
from carla_connector import CarlaConnector
import sys

def main_loop(connector):
    """
    The main loop for processing and visualization.
    """
    vehicle = connector.vehicle
    vehicle.set_autopilot(True)

    spectator = connector.world.get_spectator()

    try:
        while True:
            transform = carla.Transform(
                vehicle.get_transform().transform(carla.Location(x=-5, z=2.5)),
                vehicle.get_transform().rotation
            )
            spectator.set_transform(transform)

            image = connector.image_queue.get()

            cv2.imshow("CARLA Camera", image)

            if cv2.waitKey(1) == ord('q'):
                break
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