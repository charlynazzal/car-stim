import carla
import numpy as np
from collections import deque

class VehicleController:
    """
    Enhanced vehicle controller with obstacle avoidance and improved lane following.
    
    This controller implements:
    1. Robust PID steering control
    2. Adaptive speed control based on road conditions
    3. Basic obstacle avoidance
    4. Safety mechanisms and emergency stops
    5. Smooth control transitions
    """
    
    def __init__(self, vehicle):
        """Initialize the enhanced vehicle controller."""
        self.vehicle = vehicle
        
        # Enhanced control parameters
        self.max_steer_angle = 0.4      # Reduced for stability
        self.base_throttle = 0.35       # Conservative base speed
        self.max_throttle = 0.6         # Maximum throttle
        self.min_throttle = 0.15        # Minimum to keep moving
        
        # Improved PID parameters - more stable
        self.kp_steer = 0.0015          # Reduced proportional gain
        self.ki_steer = 0.00003         # Reduced integral gain
        self.kd_steer = 0.006           # Reduced derivative gain
        
        # Control history and smoothing
        self.error_history = deque(maxlen=8)
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_steer = 0.0
        self.prev_throttle = self.base_throttle
        
        # Enhanced smoothing parameters
        self.steer_smoothing = 0.6      # Steering smoothing factor
        self.throttle_smoothing = 0.7   # Throttle smoothing factor
        self.max_steer_change = 0.1     # Maximum steering change per frame
        
        # Safety and obstacle avoidance
        self.emergency_brake_distance = 50  # Pixels - distance to obstacle for emergency brake
        self.slow_down_distance = 100       # Pixels - distance to start slowing down
        self.obstacle_detected = False
        self.frames_since_obstacle = 0
        
        # Performance tracking
        self.frames_controlled = 0
        self.total_steering_error = 0.0
        self.max_error_seen = 0.0
        
        # Lane following state
        self.lane_following_active = False
        self.consecutive_good_frames = 0
        self.consecutive_bad_frames = 0
        
        print("Enhanced vehicle controller initialized")
        print("Features: PID steering, obstacle avoidance, adaptive speed control")
    
    def calculate_steering_angle(self, steering_error, confidence):
        """
        Calculate steering angle using enhanced PID control.
        
        Args:
            steering_error: Lane center error in pixels
            confidence: Detection confidence level
            
        Returns:
            steering_angle: Steering angle between -1.0 and 1.0
        """
        if steering_error is None:
            # No steering input - gradually return to center
            target_steer = 0.0
            return self._smooth_steering_transition(target_steer)
        
        # Safety clamp
        steering_error = np.clip(steering_error, -300, 300)
        
        # Correct steering direction
        corrected_error = -steering_error
        
        # Confidence-based gain adjustment
        confidence_multiplier = {
            'HIGH': 1.0,
            'MEDIUM': 0.8,
            'LOW': 0.6,
            'NONE': 0.3
        }
        gain_factor = confidence_multiplier.get(confidence, 0.3)
        
        # Add to error history
        self.error_history.append(corrected_error)
        
        # PID calculation
        # Proportional term
        p_term = self.kp_steer * corrected_error * gain_factor
        
        # Integral term with windup prevention
        self.integral_error += corrected_error
        self.integral_error = np.clip(self.integral_error, -300, 300)
        i_term = self.ki_steer * self.integral_error * gain_factor
        
        # Derivative term
        d_error = corrected_error - self.prev_error
        d_term = self.kd_steer * d_error * gain_factor
        
        # Calculate raw steering
        raw_steer = p_term + i_term + d_term
        
        # Apply limits
        steering_angle = np.clip(raw_steer, -self.max_steer_angle, self.max_steer_angle)
        
        # Safety check
        if not np.isfinite(steering_angle):
            steering_angle = 0.0
        
        # Update previous error
        self.prev_error = corrected_error
        
        return self._smooth_steering_transition(steering_angle)
    
    def _smooth_steering_transition(self, target_steer):
        """Apply smooth steering transitions to prevent jerky movements."""
        # Limit maximum change per frame
        steer_change = target_steer - self.prev_steer
        if abs(steer_change) > self.max_steer_change:
            direction = 1 if steer_change > 0 else -1
            target_steer = self.prev_steer + (direction * self.max_steer_change)
        
        # Apply smoothing
        smoothed_steer = (self.steer_smoothing * self.prev_steer + 
                         (1 - self.steer_smoothing) * target_steer)
        
        # Final clamp and safety check
        smoothed_steer = np.clip(smoothed_steer, -1.0, 1.0)
        if not np.isfinite(smoothed_steer):
            smoothed_steer = 0.0
        
        self.prev_steer = smoothed_steer
        return smoothed_steer
    
    def calculate_throttle_with_obstacles(self, steering_angle, confidence, steering_error, obstacles_detected=False):
        """
        Calculate throttle with obstacle awareness and adaptive speed control.
        
        Args:
            steering_angle: Current steering angle
            confidence: Detection confidence
            steering_error: Current steering error
            obstacles_detected: Whether obstacles are detected ahead
            
        Returns:
            throttle: Throttle value between 0.0 and 1.0
        """
        # Update obstacle state
        if obstacles_detected:
            self.obstacle_detected = True
            self.frames_since_obstacle = 0
        else:
            self.frames_since_obstacle += 1
            # Clear obstacle flag after 10 frames without detection
            if self.frames_since_obstacle > 10:
                self.obstacle_detected = False
        
        # Base throttle based on confidence
        confidence_throttle = {
            'HIGH': self.base_throttle,
            'MEDIUM': self.base_throttle * 0.85,
            'LOW': self.base_throttle * 0.7,
            'NONE': self.min_throttle
        }
        base_throttle = confidence_throttle.get(confidence, self.min_throttle)
        
        # Speed reduction for steering
        steer_magnitude = abs(steering_angle) if steering_angle is not None else 0
        if steer_magnitude > 0.25:
            steer_factor = 0.6  # Significant speed reduction for sharp turns
        elif steer_magnitude > 0.15:
            steer_factor = 0.75  # Moderate speed reduction
        else:
            steer_factor = 1.0  # Normal speed
        
        # Speed reduction for large errors
        error_factor = 1.0
        if steering_error is not None and abs(steering_error) > 80:
            error_factor = max(0.5, 1.0 - (abs(steering_error) - 80) / 150)
        
        # Obstacle avoidance - reduce speed or stop
        obstacle_factor = 1.0
        if self.obstacle_detected:
            obstacle_factor = 0.3  # Significant speed reduction when obstacles detected
            print("OBSTACLE DETECTED - REDUCING SPEED")
        
        # Calculate target throttle
        target_throttle = base_throttle * steer_factor * error_factor * obstacle_factor
        target_throttle = np.clip(target_throttle, self.min_throttle, self.max_throttle)
        
        # Apply smoothing
        smoothed_throttle = (self.throttle_smoothing * self.prev_throttle + 
                           (1 - self.throttle_smoothing) * target_throttle)
        
        # Final safety checks
        smoothed_throttle = np.clip(smoothed_throttle, 0.0, 1.0)
        if not np.isfinite(smoothed_throttle):
            smoothed_throttle = self.min_throttle
        
        self.prev_throttle = smoothed_throttle
        return smoothed_throttle
    
    def should_emergency_brake(self, lane_center_info, obstacles_detected):
        """
        Determine if emergency braking is needed.
        
        Args:
            lane_center_info: Lane detection information
            obstacles_detected: Whether obstacles are detected
            
        Returns:
            bool: True if emergency brake should be applied
        """
        # Emergency brake conditions
        emergency_conditions = [
            # No lane detection for extended period
            lane_center_info.get('confidence') == 'NONE' and self.consecutive_bad_frames > 20,
            
            # Very large steering error (way off lane)
            lane_center_info.get('steering_error') is not None and 
            abs(lane_center_info.get('steering_error', 0)) > 200,
            
            # Obstacles detected (simple implementation)
            obstacles_detected and self.obstacle_detected
        ]
        
        return any(emergency_conditions)
    
    def update_lane_following_state(self, lane_center_info):
        """Update the lane following state based on detection quality."""
        confidence = lane_center_info.get('confidence', 'NONE')
        
        if confidence in ['HIGH', 'MEDIUM']:
            self.consecutive_good_frames += 1
            self.consecutive_bad_frames = 0
            if self.consecutive_good_frames >= 5:
                self.lane_following_active = True
        else:
            self.consecutive_bad_frames += 1
            self.consecutive_good_frames = 0
            if self.consecutive_bad_frames >= 15:
                self.lane_following_active = False
    
    def apply_control(self, lane_center_info, obstacles_detected=False):
        """
        Apply enhanced vehicle control with obstacle avoidance.
        
        Args:
            lane_center_info: Dictionary with lane center information
            obstacles_detected: Whether obstacles are detected ahead
            
        Returns:
            control_info: Dictionary with control information
        """
        self.frames_controlled += 1
        
        # Update lane following state
        self.update_lane_following_state(lane_center_info)
        
        # Extract lane information
        steering_error = lane_center_info.get('steering_error')
        confidence = lane_center_info.get('confidence', 'NONE')
        
        # Check for emergency brake conditions
        if self.should_emergency_brake(lane_center_info, obstacles_detected):
            # Emergency stop
            control = carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=0.8,
                hand_brake=False
            )
            self.vehicle.apply_control(control)
            
            return {
                'steering_angle': 0.0,
                'throttle': 0.0,
                'brake': 0.8,
                'safety_status': 'EMERGENCY_BRAKE',
                'lane_following_active': False,
                'obstacle_detected': obstacles_detected
            }
        
        # Calculate steering and throttle
        steering_angle = self.calculate_steering_angle(steering_error, confidence)
        throttle = self.calculate_throttle_with_obstacles(
            steering_angle, confidence, steering_error, obstacles_detected
        )
        
        # Determine safety status
        if confidence == 'HIGH' and not obstacles_detected:
            safety_status = 'NORMAL'
        elif confidence in ['MEDIUM', 'LOW'] or obstacles_detected:
            safety_status = 'CAUTION'
        else:
            safety_status = 'DEGRADED'
        
        # Apply control to vehicle
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steering_angle,
            brake=0.0,
            hand_brake=False
        )
        self.vehicle.apply_control(control)
        
        # Update statistics
        if steering_error is not None:
            self.total_steering_error += abs(steering_error)
            self.max_error_seen = max(self.max_error_seen, abs(steering_error))
        
        return {
            'steering_angle': steering_angle,
            'throttle': throttle,
            'brake': 0.0,
            'safety_status': safety_status,
            'lane_following_active': self.lane_following_active,
            'obstacle_detected': self.obstacle_detected,
            'confidence': confidence,
            'frames_controlled': self.frames_controlled
        }
    
    def get_performance_stats(self):
        """Get performance statistics for monitoring."""
        if self.frames_controlled > 0:
            avg_error = self.total_steering_error / self.frames_controlled
        else:
            avg_error = 0.0
        
        return {
            'frames_controlled': self.frames_controlled,
            'average_steering_error': avg_error,
            'max_error_seen': self.max_error_seen,
            'lane_following_active': self.lane_following_active,
            'consecutive_good_frames': self.consecutive_good_frames,
            'consecutive_bad_frames': self.consecutive_bad_frames
        }
    
    def reset_controller(self):
        """Reset controller state for new session."""
        self.error_history.clear()
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_steer = 0.0
        self.prev_throttle = self.base_throttle
        self.frames_controlled = 0
        self.total_steering_error = 0.0
        self.max_error_seen = 0.0
        self.lane_following_active = False
        self.consecutive_good_frames = 0
        self.consecutive_bad_frames = 0
        self.obstacle_detected = False
        self.frames_since_obstacle = 0
        print("Vehicle controller reset") 