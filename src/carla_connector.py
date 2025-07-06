import carla
import random
import queue
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarlaConnector:
    """
    Manages the connection to a CARLA simulator for modern autonomous driving.
    
    Provides enhanced functionality for:
    - Robust connection management
    - Optimized camera setup for computer vision
    - Improved image processing pipeline
    - Better error handling and recovery
    """
    
    def __init__(self, host='localhost', port=2000):
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.image_queue = queue.Queue(maxsize=10)  # Prevent memory buildup
        self.actor_list = []
        self.host = host
        self.port = port
        self.connected = False
        self.last_image = None
        self.image_timestamp = 0
        
        # Connect automatically
        self.connect()

    def connect(self):
        """
        Connects to the CARLA simulator with enhanced error handling.
        
        Raises:
            RuntimeError: If the connection to the simulator fails.
        """
        try:
            logger.info(f"Connecting to CARLA at {self.host}:{self.port}")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            
            # Get world info
            world_map = self.world.get_map()
            weather = self.world.get_weather()
            
            logger.info(f"Connected to CARLA successfully")
            logger.info(f"Map: {world_map.name}")
            logger.info(f"Weather: Cloudiness={weather.cloudiness}, "
                       f"Precipitation={weather.precipitation}, "
                       f"Sun Altitude={weather.sun_altitude_angle}")
            
            self.connected = True
            
        except RuntimeError as e:
            logger.error(f"Error connecting to CARLA: {e}")
            self.connected = False
            raise

    def spawn_vehicle(self, blueprint_name='vehicle.tesla.model3', spawn_point=None):
        """
        Spawns a vehicle in the world with enhanced configuration.

        Args:
            blueprint_name (str): The blueprint of the vehicle to spawn.
            spawn_point (carla.Transform, optional): Specific spawn point. 
                                                     If None, a random one is chosen.

        Returns:
            carla.Actor: The spawned vehicle actor.
        """
        if not self.connected:
            raise RuntimeError("Not connected to CARLA")
            
        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find(blueprint_name)
            
            # Configure vehicle attributes for better performance
            if vehicle_bp.has_attribute('color'):
                color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                vehicle_bp.set_attribute('color', color)

            if spawn_point is None:
                map_spawn_points = self.world.get_map().get_spawn_points()
                if not map_spawn_points:
                    raise RuntimeError("No spawn points available")
                spawn_point = random.choice(map_spawn_points)

            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.actor_list.append(self.vehicle)
            
            logger.info(f"Spawned vehicle: {self.vehicle.type_id} at {spawn_point.location}")
            return self.vehicle
            
        except Exception as e:
            logger.error(f"Error spawning vehicle: {e}")
            raise

    def setup_camera(self, attach_to, width=640, height=480, fov=110):
        """
        Creates and attaches an RGB camera sensor optimized for computer vision.
        
        Optimized settings for modern CV:
        - Standard resolution (640x480) for good performance
        - Wide FOV (110°) to capture more road context
        - Positioned for optimal lane and object detection
        - Enhanced image processing pipeline

        Args:
            attach_to (carla.Actor): The actor to attach the camera to.
            width (int): The width of the camera image.
            height (int): The height of the camera image.
            fov (float): The field of view of the camera.

        Returns:
            carla.Actor: The spawned camera sensor actor.
        """
        if not self.connected:
            raise RuntimeError("Not connected to CARLA")
            
        try:
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(width))
            camera_bp.set_attribute('image_size_y', str(height))
            camera_bp.set_attribute('fov', str(fov))
            
            # Enhanced camera settings for computer vision
            camera_bp.set_attribute('sensor_tick', '0.033')  # ~30 FPS
            camera_bp.set_attribute('gamma', '2.2')
            camera_bp.set_attribute('motion_blur_intensity', '0.0')
            camera_bp.set_attribute('motion_blur_max_distortion', '0.0')

            # Optimized camera position for lane detection and object detection
            # Positioned to get good view of road ahead and surrounding objects
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),  # Forward and up
                carla.Rotation(pitch=-10.0)     # Slight downward angle
            )
            
            self.camera = self.world.spawn_actor(
                camera_bp, 
                camera_transform, 
                attach_to=attach_to
            )
            self.actor_list.append(self.camera)
            self.camera.listen(self._process_image)
            
            logger.info(f"Camera attached to {attach_to.type_id}")
            logger.info(f"Camera settings: {width}x{height}, FOV={fov}°")
            
            return self.camera
            
        except Exception as e:
            logger.error(f"Error setting up camera: {e}")
            raise

    def _process_image(self, image):
        """
        Enhanced callback function to process images from the camera sensor.
        
        Processes images for modern computer vision pipeline:
        - Converts from CARLA format to OpenCV format
        - Handles color space conversion
        - Implements queue management to prevent memory issues
        - Stores latest image for immediate access
        """
        try:
            # Convert CARLA image to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            
            # Convert RGBA to BGR for OpenCV compatibility
            bgr_image = array[:, :, :3]  # Remove alpha channel
            bgr_image = bgr_image[:, :, ::-1]  # RGB to BGR
            
            # Store latest image for immediate access
            self.last_image = bgr_image.copy()
            self.image_timestamp = time.time()
            
            # Add to queue with overflow protection
            try:
                self.image_queue.put_nowait(bgr_image)
            except queue.Full:
                # Remove oldest image and add new one
                try:
                    self.image_queue.get_nowait()
                    self.image_queue.put_nowait(bgr_image)
                except queue.Empty:
                    pass
                    
        except Exception as e:
            logger.error(f"Error processing image: {e}")

    def get_camera_image(self):
        """
        Get the latest camera image.
        
        Returns:
            numpy.ndarray: Latest camera image in BGR format, or None if no image available
        """
        # Try to get from queue first (most recent)
        try:
            return self.image_queue.get_nowait()
        except queue.Empty:
            # Fallback to stored image
            if self.last_image is not None:
                # Check if image is recent (within 1 second)
                if time.time() - self.image_timestamp < 1.0:
                    return self.last_image.copy()
            return None

    def get_world_info(self):
        """Get information about the current world."""
        if not self.connected:
            return None
            
        try:
            world_map = self.world.get_map()
            weather = self.world.get_weather()
            
            return {
                'map_name': world_map.name,
                'weather': {
                    'cloudiness': weather.cloudiness,
                    'precipitation': weather.precipitation,
                    'sun_altitude': weather.sun_altitude_angle,
                    'wind_intensity': weather.wind_intensity
                },
                'actors_count': len(self.world.get_actors()),
                'spawn_points': len(world_map.get_spawn_points())
            }
        except Exception as e:
            logger.error(f"Error getting world info: {e}")
            return None

    def set_weather(self, weather_preset='ClearNoon'):
        """
        Set weather conditions for optimal computer vision.
        
        Args:
            weather_preset (str): Weather preset name
        """
        if not self.connected:
            return
            
        try:
            weather_presets = {
                'ClearNoon': carla.WeatherParameters.ClearNoon,
                'CloudyNoon': carla.WeatherParameters.CloudyNoon,
                'WetNoon': carla.WeatherParameters.WetNoon,
                'ClearSunset': carla.WeatherParameters.ClearSunset,
                'CloudySunset': carla.WeatherParameters.CloudySunset,
                'WetSunset': carla.WeatherParameters.WetSunset
            }
            
            if weather_preset in weather_presets:
                self.world.set_weather(weather_presets[weather_preset])
                logger.info(f"Weather set to {weather_preset}")
            else:
                logger.warning(f"Unknown weather preset: {weather_preset}")
                
        except Exception as e:
            logger.error(f"Error setting weather: {e}")

    def get_vehicle_status(self):
        """Get current vehicle status."""
        if not self.vehicle:
            return None
            
        try:
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            
            return {
                'location': {
                    'x': transform.location.x,
                    'y': transform.location.y,
                    'z': transform.location.z
                },
                'rotation': {
                    'pitch': transform.rotation.pitch,
                    'yaw': transform.rotation.yaw,
                    'roll': transform.rotation.roll
                },
                'speed': speed,
                'velocity': {
                    'x': velocity.x,
                    'y': velocity.y,
                    'z': velocity.z
                }
            }
        except Exception as e:
            logger.error(f"Error getting vehicle status: {e}")
            return None

    def cleanup(self):
        """
        Enhanced cleanup with better error handling.
        """
        logger.info("Starting cleanup...")
        
        # Stop camera listening
        if self.camera and hasattr(self.camera, 'is_listening') and self.camera.is_listening:
            try:
                self.camera.stop()
                logger.info("Camera stopped")
            except Exception as e:
                logger.error(f"Error stopping camera: {e}")

        # Destroy all actors
        destroyed_count = 0
        for actor in self.actor_list:
            try:
                if hasattr(actor, 'is_alive') and actor.is_alive:
                    actor.destroy()
                    destroyed_count += 1
            except Exception as e:
                logger.error(f"Error destroying actor: {e}")
        
        self.actor_list.clear()
        
        # Clear image queue
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset state
        self.vehicle = None
        self.camera = None
        self.last_image = None
        self.connected = False
        
        logger.info(f"Cleanup complete. Destroyed {destroyed_count} actors")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass 