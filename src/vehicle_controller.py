import carla
import numpy as np
from collections import deque

class VehicleController:
    """
    Controls vehicle steering and throttle based on lane detection results.
    
    This class implements:
    1. Proportional steering control based on lane center error
    2. Adaptive speed control that slows down for sharp turns
    3. Safety mechanisms and control limits
    4. Smooth control transitions to prevent jerky movements
    """
    
    def __init__(self, vehicle):
        """
        Initialize the vehicle controller.
        
        Args:
            vehicle: CARLA vehicle actor to control
        """
        self.vehicle = vehicle
        
        # Control parameters
        self.max_steer_angle = 0.5      # Maximum steering angle (-0.5 to 0.5)
        self.base_throttle = 0.4        # Base throttle for normal driving (increased from 0.3)
        self.max_throttle = 0.7         # Maximum throttle (increased from 0.6)
        self.min_throttle = 0.2         # Minimum throttle to keep moving (increased from 0.1)
        
        # PID Controller parameters for steering
        self.kp_steer = 0.005           # Proportional gain for steering (reduced for smoother control)
        self.ki_steer = 0.0001          # Integral gain for steering (reduced for stability)
        self.kd_steer = 0.015           # Derivative gain for steering (reduced for less oscillation)
        
        # Control history for derivative and integral terms
        self.error_history = deque(maxlen=10)
        self.integral_error = 0.0
        self.prev_error = 0.0
        
        # Smoothing parameters
        self.control_smoothing = 0.7    # Smoothing factor
        self.prev_steer = 0.0
        self.prev_throttle = self.base_throttle
        
        # Safety parameters
        self.max_error_threshold = 200  # Maximum acceptable steering error (pixels)
        self.emergency_brake_threshold = 300  # Emergency brake threshold (pixels)
        
        # Performance tracking
        self.control_active = False
        self.frames_controlled = 0
        self.initial_boost_frames = 30  # Apply extra throttle for first 30 frames
        
        print("Vehicle controller initialized with PID steering control")
    
    def calculate_steering_angle(self, steering_error, confidence):
        """
        Calculate steering angle using PID control with enhanced stability.
        
        Args:
            steering_error: Lane center error in pixels (positive = steer left)
            confidence: Detection confidence level ('HIGH', 'MEDIUM', 'LOW', 'NONE')
            
        Returns:
            steering_angle: Steering angle between -1.0 and 1.0
        """
        if steering_error is None:
            return self.prev_steer  # Maintain previous steering if no input
        
        # Safety check: clamp steering error to reasonable range
        steering_error = np.clip(steering_error, -400, 400)
        
        # Use steering error directly (positive error = steer right, negative = steer left)
        corrected_error = steering_error
        
        # Confidence-based gain adjustment
        confidence_multiplier = {
            'HIGH': 1.0,
            'MEDIUM': 0.8,
            'LOW': 0.5,
            'NONE': 0.0
        }
        
        gain_factor = confidence_multiplier.get(confidence, 0.0)
        
        # Add corrected error to history
        self.error_history.append(corrected_error)
        
        # Calculate PID terms using corrected error
        # Proportional term
        p_term = self.kp_steer * corrected_error * gain_factor
        
        # Integral term (accumulated error over time) with windup prevention
        self.integral_error += corrected_error
        # More aggressive integral windup prevention
        self.integral_error = np.clip(self.integral_error, -500, 500)
        i_term = self.ki_steer * self.integral_error * gain_factor
        
        # Derivative term (rate of change of error)
        d_error = corrected_error - self.prev_error
        d_term = self.kd_steer * d_error * gain_factor
        
        # Calculate raw steering angle
        raw_steer = p_term + i_term + d_term
        
        # Apply steering limits
        steering_angle = np.clip(raw_steer, -self.max_steer_angle, self.max_steer_angle)
        
        # Safety check: ensure steering angle is valid
        if not np.isfinite(steering_angle):
            steering_angle = 0.0
        
        # Apply smoothing to prevent jerky movements
        smoothed_steer = (self.control_smoothing * self.prev_steer + 
                         (1 - self.control_smoothing) * steering_angle)
        
        # Final safety check on smoothed value
        smoothed_steer = np.clip(smoothed_steer, -1.0, 1.0)
        if not np.isfinite(smoothed_steer):
            smoothed_steer = 0.0
        
        # Update previous values
        self.prev_error = corrected_error
        self.prev_steer = smoothed_steer
        
        return smoothed_steer
    
    def calculate_throttle(self, steering_angle, confidence, steering_error):
        """
        Calculate throttle based on steering requirements and confidence with enhanced stability.
        
        Args:
            steering_angle: Current steering angle
            confidence: Detection confidence level
            steering_error: Current steering error for adaptive speed
            
        Returns:
            throttle: Throttle value between 0.0 and 1.0
        """
        # Safety check: ensure steering angle is valid
        if not np.isfinite(steering_angle):
            steering_angle = 0.0
        
        # Base throttle adjustment based on confidence
        confidence_throttle = {
            'HIGH': self.base_throttle,
            'MEDIUM': self.base_throttle * 0.8,
            'LOW': self.base_throttle * 0.6,
            'NONE': self.min_throttle
        }
        
        base_throttle = confidence_throttle.get(confidence, self.min_throttle)
        
        # Reduce speed for sharp turns (adaptive speed control)
        steer_magnitude = abs(steering_angle)
        if steer_magnitude > 0.3:
            # Sharp turn - reduce speed significantly
            speed_factor = 0.5
        elif steer_magnitude > 0.15:
            # Moderate turn - reduce speed moderately
            speed_factor = 0.7
        else:
            # Straight or gentle turn - normal speed
            speed_factor = 1.0
        
        # Additional speed reduction for large steering errors
        if steering_error is not None and abs(steering_error) > 100:
            error_factor = max(0.3, 1.0 - (abs(steering_error) - 100) / 200)
        else:
            error_factor = 1.0
        
        # Calculate target throttle
        target_throttle = base_throttle * speed_factor * error_factor
        target_throttle = np.clip(target_throttle, self.min_throttle, self.max_throttle)
        
        # Apply initial boost to overcome inertia (first 30 frames)
        if self.frames_controlled < self.initial_boost_frames:
            boost_factor = 1.5 - (self.frames_controlled / self.initial_boost_frames) * 0.5
            target_throttle *= boost_factor
            target_throttle = np.clip(target_throttle, self.min_throttle, self.max_throttle)
        
        # Apply smoothing
        smoothed_throttle = (self.control_smoothing * self.prev_throttle + 
                           (1 - self.control_smoothing) * target_throttle)
        
        # Final safety checks
        smoothed_throttle = np.clip(smoothed_throttle, 0.0, 1.0)
        if not np.isfinite(smoothed_throttle):
            smoothed_throttle = self.min_throttle
        
        self.prev_throttle = smoothed_throttle
        
        return smoothed_throttle
    
    def apply_control(self, lane_center_info):
        """
        Apply vehicle control based on lane center information.
        
        Args:
            lane_center_info: Dictionary containing lane detection results
            
        Returns:
            control_info: Dictionary with applied control values
        """
        # Extract lane center information
        steering_error = lane_center_info.get('steering_error')
        confidence = lane_center_info.get('confidence', 'NONE')
        steering_direction = lane_center_info.get('steering_direction', 'UNKNOWN')
        
        # Initialize control info
        control_info = {
            'steering_angle': 0.0,
            'throttle': 0.0,
            'brake': 0.0,
            'control_active': False,
            'safety_status': 'NORMAL'
        }
        
        # Safety check - emergency brake for extreme errors
        if steering_error is not None and abs(steering_error) > self.emergency_brake_threshold:
            control_info.update({
                'steering_angle': 0.0,
                'throttle': 0.0,
                'brake': 1.0,
                'control_active': False,
                'safety_status': 'EMERGENCY_BRAKE'
            })
            
            # Apply emergency brake with error handling
            try:
                control = carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0
                )
                self.vehicle.apply_control(control)
            except Exception as e:
                print(f"Error applying emergency brake: {e}")
                control_info['safety_status'] = 'CONTROL_ERROR'
            
            return control_info
        
        # Normal control calculation
        if steering_error is not None:
            # Calculate steering angle
            steering_angle = self.calculate_steering_angle(steering_error, confidence)
            
            # Calculate throttle
            throttle = self.calculate_throttle(steering_angle, confidence, steering_error)
            
            # Final validation of control values
            steering_angle = np.clip(steering_angle, -1.0, 1.0)
            throttle = np.clip(throttle, 0.0, 1.0)
            
            # Check for invalid values
            if not (np.isfinite(steering_angle) and np.isfinite(throttle)):
                print("Warning: Invalid control values detected, using safe defaults")
                steering_angle = 0.0
                throttle = self.min_throttle
            
            # Apply control to vehicle with error handling
            try:
                control = carla.VehicleControl(
                    throttle=float(throttle),
                    steer=float(steering_angle),
                    brake=0.0,  # Explicitly set brake to 0
                    hand_brake=False  # Ensure handbrake is off
                )
                self.vehicle.apply_control(control)
                
                # Update control info
                control_info.update({
                    'steering_angle': steering_angle,
                    'throttle': throttle,
                    'brake': 0.0,
                    'control_active': True,
                    'safety_status': 'NORMAL'
                })
                
                self.control_active = True
                self.frames_controlled += 1
                
            except Exception as e:
                print(f"Error applying vehicle control: {e}")
                control_info.update({
                    'steering_angle': 0.0,
                    'throttle': 0.0,
                    'brake': 0.5,
                    'control_active': False,
                    'safety_status': 'CONTROL_ERROR'
                })
            
        else:
            # No valid lane detection - stop vehicle safely
            try:
                control = carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=0.5
                )
                self.vehicle.apply_control(control)
                
                control_info.update({
                    'steering_angle': 0.0,
                    'throttle': 0.0,
                    'brake': 0.5,
                    'control_active': False,
                    'safety_status': 'NO_LANES_DETECTED'
                })
                
            except Exception as e:
                print(f"Error applying safe stop: {e}")
                control_info['safety_status'] = 'CONTROL_ERROR'
                
            self.control_active = False
        
        return control_info
    
    def get_control_stats(self):
        """
        Get controller performance statistics.
        
        Returns:
            stats: Dictionary with performance metrics
        """
        return {
            'control_active': self.control_active,
            'frames_controlled': self.frames_controlled,
            'integral_error': self.integral_error,
            'prev_steer': self.prev_steer,
            'prev_throttle': self.prev_throttle
        }
    
    def reset_controller(self):
        """Reset controller state for new driving session."""
        self.error_history.clear()
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_steer = 0.0
        self.prev_throttle = self.base_throttle
        self.control_active = False
        self.frames_controlled = 0
        
        print("Vehicle controller reset") 