import carla
import cv2
import sys
import time

def test_carla_connection():
    """
    Simple test to check CARLA connection without complex control logic.
    """
    client = None
    vehicle = None
    camera = None
    
    try:
        print("Testing CARLA connection...")
        
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # Get world
        world = client.get_world()
        print(f"Connected to CARLA world: {world.get_map().name}")
        
        # Get blueprint library
        blueprint_library = world.get_blueprint_library()
        
        # Spawn vehicle
        vehicle_bp = blueprint_library.filter('vehicle.lincoln.mkz')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned vehicle: {vehicle.type_id}")
        
        # Test simple movement
        print("Testing basic vehicle movement...")
        vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))
        time.sleep(2)
        
        # Stop vehicle
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        print("Vehicle stopped successfully")
        
        # Test camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        print("Camera spawned successfully")
        
        # Test for 5 seconds
        print("Testing for 5 seconds... Press Ctrl+C to stop")
        start_time = time.time()
        while time.time() - start_time < 5.0:
            time.sleep(0.1)
            
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
        
    finally:
        # Cleanup
        if camera:
            camera.destroy()
        if vehicle:
            vehicle.destroy()
        print("Cleanup completed")

if __name__ == '__main__':
    success = test_carla_connection()
    sys.exit(0 if success else 1) 