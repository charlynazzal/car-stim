import carla

def list_available_vehicles():
    """
    Connect to CARLA and list all available vehicle blueprints.
    This helps us find the correct vehicle names for our CARLA version.
    """
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # Get blueprint library
        blueprint_library = world.get_blueprint_library()
        
        # Filter for vehicles only
        vehicles = blueprint_library.filter('vehicle.*')
        
        print("Available vehicles in your CARLA installation:")
        print("=" * 50)
        
        for vehicle in vehicles:
            print(f"- {vehicle.id}")
            
        print(f"\nTotal vehicles found: {len(vehicles)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    list_available_vehicles() 