import carla
import random
import queue
import numpy as np

class CarlaConnector:
    """
    Manages the connection to a CARLA simulator, vehicle spawning,
    and sensor setup.
    """
    def __init__(self, host='localhost', port=2000):
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.image_queue = queue.Queue()
        self.actor_list = []
        self.host = host
        self.port = port

    def connect(self):
        """
        Connects to the CARLA simulator.
        
        Raises:
            RuntimeError: If the connection to the simulator fails.
        """
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            print("Successfully connected to CARLA.")
        except RuntimeError as e:
            print(f"Error connecting to CARLA: {e}")
            raise

    def spawn_vehicle(self, blueprint_name='vehicle.tesla.model3', spawn_point=None):
        """
        Spawns a vehicle in the world.

        Args:
            blueprint_name (str): The blueprint of the vehicle to spawn.
            spawn_point (carla.Transform, optional): Specific spawn point. 
                                                     If None, a random one is chosen.

        Returns:
            carla.Actor: The spawned vehicle actor.
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(blueprint_name)

        if spawn_point is None:
            map_spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(map_spawn_points) if map_spawn_points else carla.Transform()

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)
        print(f"Spawned vehicle: {self.vehicle.type_id}")
        return self.vehicle

    def setup_camera(self, attach_to, width=800, height=600, fov=90):
        """
        Creates and attaches an RGB camera sensor to an actor.

        Args:
            attach_to (carla.Actor): The actor to attach the camera to.
            width (int): The width of the camera image.
            height (int): The height of the camera image.
            fov (float): The field of view of the camera.

        Returns:
            carla.Actor: The spawned camera sensor actor.
        """
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        camera_bp.set_attribute('fov', str(fov))

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        
        self.camera = self.world.spawn_actor(
            camera_bp, 
            camera_transform, 
            attach_to=attach_to
        )
        self.actor_list.append(self.camera)
        self.camera.listen(self._process_image)
        print(f"Spawned camera and attached to {attach_to.type_id}")
        return self.camera

    def _process_image(self, image):
        """
        Callback function to process images from the camera sensor and
        add them to a thread-safe queue.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        bgr_image = array[:, :, :3]
        self.image_queue.put(bgr_image)

    def cleanup(self):
        """
        Destroys all spawned actors to clean up the simulation environment.
        """
        print("Destroying actors...")
        if self.camera and self.camera.is_listening:
            self.camera.stop()

        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        self.actor_list.clear()
        print("All actors destroyed.") 