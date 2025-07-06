import carla
import numpy as np
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModernVehicleController:
    """
    Modern vehicle controller for autonomous driving.
    
    Integrates with YOLOv8 object detection and semantic segmentation
    for intelligent decision making and safe vehicle control.
    """
    
    def __init__(self, vehicle):
        """Initialize the modern vehicle controller."""
        self.vehicle = vehicle
        
        # PID parameters for steering (optimized for modern CV input)
        self.kp_steer = 0.003
        self.ki_steer = 0.0001
        self.kd_steer = 0.008
        
        # PID state
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_steering = 0.0
        
        # Speed control parameters
        self.base_speed = 35.0  # km/h
        self.max_speed = 50.0
        self.min_speed = 10.0
        
        # Safety parameters
        self.emergency_brake_threshold = 0.3
        self.speed_reduction_threshold = 0.6
        self.object_distance_threshold = 50.0  # pixels
        
        # Adaptive control parameters
        self.confidence_threshold = 0.7
        self.steering_smoothing = 0.85
        self.throttle_smoothing = 0.9
        
        # State tracking
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.control_history = deque(maxlen=10)
        
        # Performance metrics
        self.control_times = deque(maxlen=50)
        
        logger.info("Modern vehicle controller initialized")
    
    def apply_control(self, cv_results):
        """
        Apply vehicle control based on modern computer vision results.
        
        Args:
            cv_results: Results from ModernLaneDetector containing:
                - objects: Object detection results
                - lanes: Lane segmentation results  
                - safety: Safety assessment
                - lane_center: Detected lane center
                - lane_confidence: Confidence in lane detection
                - steering_error: Error from lane center
        """
        import time
        start_time = time.time()
        
        try:
            # Extract information from CV results
            objects = cv_results.get('objects', {})
            lanes = cv_results.get('lanes', {})
            safety = cv_results.get('safety', {})
            lane_center = cv_results.get('lane_center', 320)
            lane_confidence = cv_results.get('lane_confidence', 0.5)
            steering_error = cv_results.get('steering_error', 0)
            
            # Calculate steering control
            steering = self._calculate_steering(steering_error, lane_confidence)
            
            # Calculate speed control based on safety and objects
            target_speed = self._calculate_target_speed(safety, objects, lane_confidence)
            
            # Calculate throttle and brake
            throttle, brake = self._calculate_throttle_brake(target_speed, safety)
            
            # Apply smoothing
            steering = self._smooth_steering(steering)
            throttle = self._smooth_throttle(throttle)
            brake = self._smooth_brake(brake)
            
            # Create and apply control
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steering,
                brake=brake,
                hand_brake=False,
                reverse=False
            )
            
            # Apply control to vehicle
            self.vehicle.apply_control(control)
            
            # Record control for analysis
            self._record_control(control, cv_results)
            
            # Record processing time
            control_time = time.time() - start_time
            self.control_times.append(control_time)
            
            return {
                'steering': steering,
                'throttle': throttle,
                'brake': brake,
                'target_speed': target_speed,
                'control_time': control_time,
                'safety_override': safety.get('emergency_brake', False)
            }
            
        except Exception as e:
            logger.error(f"Error in vehicle control: {e}")
            # Emergency stop
            emergency_control = carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=True
            )
            self.vehicle.apply_control(emergency_control)
            return {'error': str(e)}
    
    def _calculate_steering(self, steering_error, lane_confidence):
        """Calculate steering control using PID with confidence weighting."""
        # Adjust PID gains based on confidence
        confidence_factor = max(0.3, lane_confidence)
        
        # PID calculation
        self.integral_error += steering_error * confidence_factor
        self.integral_error = np.clip(self.integral_error, -100, 100)  # Prevent windup
        
        derivative_error = steering_error - self.prev_error
        
        # PID output
        steering_output = (
            self.kp_steer * steering_error * confidence_factor +
            self.ki_steer * self.integral_error +
            self.kd_steer * derivative_error
        )
        
        # Limit steering
        steering = np.clip(steering_output, -1.0, 1.0)
        
        # Update previous error
        self.prev_error = steering_error
        
        return steering
    
    def _calculate_target_speed(self, safety, objects, lane_confidence):
        """Calculate target speed based on safety and detected objects."""
        # Start with base speed
        target_speed = self.base_speed
        
        # Reduce speed based on safety score
        safety_score = safety.get('safety_score', 1.0)
        if safety_score < self.speed_reduction_threshold:
            speed_reduction = (self.speed_reduction_threshold - safety_score) / self.speed_reduction_threshold
            target_speed *= (1.0 - speed_reduction * 0.7)
        
        # Reduce speed based on lane confidence
        if lane_confidence < self.confidence_threshold:
            confidence_reduction = (self.confidence_threshold - lane_confidence) / self.confidence_threshold
            target_speed *= (1.0 - confidence_reduction * 0.5)
        
        # Reduce speed for nearby objects
        target_speed = self._adjust_speed_for_objects(target_speed, objects)
        
        # Limit speed
        target_speed = np.clip(target_speed, self.min_speed, self.max_speed)
        
        return target_speed
    
    def _adjust_speed_for_objects(self, target_speed, objects):
        """Adjust speed based on detected objects."""
        # Check for vehicles ahead
        for vehicle in objects.get('vehicles', []):
            bbox = vehicle['bbox']
            # Check if vehicle is in front (bottom half of image)
            if bbox[1] > 240:  # Vehicle in lower half
                # Calculate approximate distance (simple heuristic)
                vehicle_height = bbox[3] - bbox[1]
                if vehicle_height > 100:  # Close vehicle
                    target_speed *= 0.6
                elif vehicle_height > 50:  # Medium distance
                    target_speed *= 0.8
        
        # Check for pedestrians
        for pedestrian in objects.get('pedestrians', []):
            bbox = pedestrian['bbox']
            # Pedestrians require more caution
            if bbox[1] > 200:  # Pedestrian visible
                target_speed *= 0.4
        
        # Check for traffic signs
        for sign in objects.get('traffic_signs', []):
            # Assume traffic signs require speed reduction
            target_speed *= 0.7
        
        return target_speed
    
    def _calculate_throttle_brake(self, target_speed, safety):
        """Calculate throttle and brake values."""
        # Get current speed
        velocity = self.vehicle.get_velocity()
        current_speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s to km/h
        
        # Emergency brake check
        if safety.get('emergency_brake', False):
            return 0.0, 1.0
        
        # Speed control
        speed_error = target_speed - current_speed
        
        if speed_error > 2.0:
            # Need to accelerate
            throttle = min(0.8, speed_error / 20.0)
            brake = 0.0
        elif speed_error < -2.0:
            # Need to brake
            throttle = 0.0
            brake = min(0.8, abs(speed_error) / 20.0)
        else:
            # Maintain speed
            throttle = 0.3
            brake = 0.0
        
        return throttle, brake
    
    def _smooth_steering(self, steering):
        """Apply smoothing to steering control."""
        smoothed_steering = (
            self.steering_smoothing * self.prev_steering +
            (1 - self.steering_smoothing) * steering
        )
        
        # Limit steering change rate
        max_change = 0.1
        steering_change = smoothed_steering - self.prev_steering
        if abs(steering_change) > max_change:
            smoothed_steering = self.prev_steering + np.sign(steering_change) * max_change
        
        self.prev_steering = smoothed_steering
        return smoothed_steering
    
    def _smooth_throttle(self, throttle):
        """Apply smoothing to throttle control."""
        smoothed_throttle = (
            self.throttle_smoothing * self.prev_throttle +
            (1 - self.throttle_smoothing) * throttle
        )
        
        self.prev_throttle = smoothed_throttle
        return smoothed_throttle
    
    def _smooth_brake(self, brake):
        """Apply smoothing to brake control."""
        smoothed_brake = (
            self.throttle_smoothing * self.prev_brake +
            (1 - self.throttle_smoothing) * brake
        )
        
        self.prev_brake = smoothed_brake
        return smoothed_brake
    
    def _record_control(self, control, cv_results):
        """Record control actions for analysis."""
        control_record = {
            'timestamp': time.time(),
            'steering': control.steer,
            'throttle': control.throttle,
            'brake': control.brake,
            'lane_confidence': cv_results.get('lane_confidence', 0),
            'safety_score': cv_results.get('safety', {}).get('safety_score', 1.0),
            'object_count': sum(len(objects) for objects in cv_results.get('objects', {}).values())
        }
        
        self.control_history.append(control_record)
    
    def get_performance_metrics(self):
        """Get controller performance metrics."""
        if not self.control_history:
            return {}
        
        recent_controls = list(self.control_history)[-10:]
        
        return {
            'avg_control_time': np.mean(self.control_times) if self.control_times else 0,
            'avg_steering': np.mean([c['steering'] for c in recent_controls]),
            'avg_throttle': np.mean([c['throttle'] for c in recent_controls]),
            'avg_brake': np.mean([c['brake'] for c in recent_controls]),
            'avg_lane_confidence': np.mean([c['lane_confidence'] for c in recent_controls]),
            'avg_safety_score': np.mean([c['safety_score'] for c in recent_controls]),
            'control_frequency': len(self.control_times) / (self.control_times[-1] - self.control_times[0] + 1e-6) if len(self.control_times) > 1 else 0
        }
    
    def reset_controller(self):
        """Reset controller state."""
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.control_history.clear()
        self.control_times.clear()
        
        logger.info("Vehicle controller reset")
    
    def update_parameters(self, **kwargs):
        """Update controller parameters dynamically."""
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
                logger.info(f"Updated {param} to {value}")
    
    def get_status(self):
        """Get current controller status."""
        return {
            'is_active': True,
            'control_mode': 'modern_cv',
            'avg_control_time': np.mean(self.control_times) if self.control_times else 0,
            'controls_applied': len(self.control_history),
            'current_steering': self.prev_steering,
            'current_throttle': self.prev_throttle,
            'current_brake': self.prev_brake
        } 