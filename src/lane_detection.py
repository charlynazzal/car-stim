import cv2
import numpy as np
from collections import deque

class LaneDetector:
    """
    Advanced lane detection system for autonomous driving with safety-first approach.
    
    This class implements:
    1. Proper ROI that connects to frame edges
    2. Lane divider detection to prevent illegal lane changes
    3. Sidewalk classification to prevent sidewalk approach
    4. Strict lane boundary enforcement
    5. Enhanced color detection for road markings
    """
    
    def __init__(self):
        """Initialize the enhanced lane detector with safety-focused parameters."""
        # Image dimensions (should match camera settings)
        self.image_width = 640
        self.image_height = 480
        
        # FIXED: ROI trapezoid that connects to frame edges
        # This ensures we capture all lane markings at the frame boundaries
        self.roi_vertices = np.array([
            [0, self.image_height - 80],                    # Bottom left - connected to edge
            [self.image_width // 2 - 120, self.image_height // 2 - 60],  # Top left - wider for distant detection
            [self.image_width // 2 + 120, self.image_height // 2 - 60],  # Top right - wider for distant detection
            [self.image_width - 1, self.image_height - 80]  # Bottom right - connected to edge
        ], dtype=np.int32)
        
        # Enhanced temporal filtering for stability
        self.history_size = 8  # Optimized history size
        self.lane_center_history = deque(maxlen=self.history_size)
        self.left_lane_history = deque(maxlen=self.history_size)
        self.right_lane_history = deque(maxlen=self.history_size)
        self.lane_divider_history = deque(maxlen=self.history_size)
        
        # Safety boundaries for lane enforcement
        self.road_left_boundary = self.image_width * 0.25   # 25% from left edge
        self.road_right_boundary = self.image_width * 0.75  # 75% from right edge
        self.lane_center_safe_zone = (self.image_width * 0.40, self.image_width * 0.60)  # Safe driving zone
        
        # Detection confidence tracking
        self.min_confidence_lines = 2
        self.outlier_threshold = 60  # Balanced threshold
        
        # Previous frame memory for validation
        self.prev_left_lane_x = None
        self.prev_right_lane_x = None
        self.prev_lane_center_x = None
        self.prev_lane_divider_x = None
        self.prev_steering_direction = 'STRAIGHT'
        
        # Safety state tracking
        self.sidewalk_detected = False
        self.lane_divider_detected = False
        self.illegal_lane_change_risk = False
        
        print("Enhanced lane detector initialized with safety-first approach")
        print(f"ROI connects to frame edges: (0,{self.image_height-80}) to ({self.image_width-1},{self.image_height-80})")
    
    def preprocess_image(self, image):
        """
        Enhanced preprocessing with multi-target detection for lanes, dividers, and sidewalks.
        
        Args:
            image: RGB image from CARLA camera
            
        Returns:
            Dictionary containing different processed images for various detection targets
        """
        # Convert RGB to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 1. WHITE LANE MARKINGS (dashed lines on road edges)
        white_lower = np.array([0, 0, 180])     # Bright white with some tolerance
        white_upper = np.array([180, 40, 255])  # Allow slight color variation
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # 2. YELLOW LANE DIVIDERS (center lines separating opposing traffic)
        yellow_lower = np.array([20, 100, 100])  # True yellow range
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # 3. GRAY SIDEWALKS (concrete/asphalt sidewalks)
        gray_lower = np.array([0, 0, 80])       # Dark gray
        gray_upper = np.array([180, 50, 150])   # Light gray
        gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
        
        # Create combined lane mask (white + yellow)
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Apply morphological operations to clean up masks
        kernel = np.ones((3,3), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
        
        # Create processed images for different detection targets
        lane_image = cv2.bitwise_and(image, image, mask=lane_mask)
        sidewalk_image = cv2.bitwise_and(image, image, mask=gray_mask)
        
        # Convert to grayscale for edge detection
        lane_gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
        sidewalk_gray = cv2.cvtColor(sidewalk_image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        lane_blurred = cv2.GaussianBlur(lane_gray, (5, 5), 0)
        sidewalk_blurred = cv2.GaussianBlur(sidewalk_gray, (7, 7), 0)
        
        # Canny edge detection with optimized parameters
        lane_edges = cv2.Canny(lane_blurred, 50, 150)
        sidewalk_edges = cv2.Canny(sidewalk_blurred, 30, 100)
        
        return {
            'lane_edges': lane_edges,
            'sidewalk_edges': sidewalk_edges,
            'white_mask': white_mask,
            'yellow_mask': yellow_mask,
            'gray_mask': gray_mask,
            'combined_edges': cv2.bitwise_or(lane_edges, sidewalk_edges)
        }
    
    def apply_region_of_interest(self, image):
        """
        Apply ROI mask that connects to frame edges for complete lane detection.
        
        Args:
            image: Edge-detected image
            
        Returns:
            masked_image: Image with ROI applied
        """
        # Create mask
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [self.roi_vertices], 255)
        
        # Apply mask
        masked_image = cv2.bitwise_and(image, mask)
        
        return masked_image
    
    def detect_sidewalks(self, sidewalk_edges):
        """
        Detect sidewalks to prevent vehicle from approaching them.
        
        Args:
            sidewalk_edges: Edge-detected sidewalk image
            
        Returns:
            sidewalk_boundaries: List of detected sidewalk boundaries
        """
        # Apply ROI to sidewalk detection
        masked_sidewalk = self.apply_region_of_interest(sidewalk_edges)
        
        # Detect lines in sidewalk areas
        lines = cv2.HoughLinesP(
            masked_sidewalk,
            rho=1,
            theta=np.pi/180,
            threshold=40,
            minLineLength=50,
            maxLineGap=20
        )
        
        sidewalk_boundaries = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line is horizontal (sidewalk edge)
                if abs(y2 - y1) < 20:  # Nearly horizontal
                    line_center_x = (x1 + x2) / 2
                    
                    # Check if it's at the edges (likely sidewalk)
                    if line_center_x < self.image_width * 0.3 or line_center_x > self.image_width * 0.7:
                        sidewalk_boundaries.append(line)
        
        self.sidewalk_detected = len(sidewalk_boundaries) > 0
        return sidewalk_boundaries
    
    def detect_lane_dividers(self, yellow_mask):
        """
        Detect yellow lane dividers that separate opposing traffic lanes.
        
        Args:
            yellow_mask: Binary mask of yellow pixels
            
        Returns:
            lane_divider_x: X-coordinate of detected lane divider, or None
        """
        # Apply ROI to yellow mask
        masked_yellow = self.apply_region_of_interest(yellow_mask)
        
        # Find contours in yellow mask
        contours, _ = cv2.findContours(masked_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lane_divider_candidates = []
        
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            if area < 100:  # Too small to be a lane divider
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's vertical (lane divider characteristic)
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio > 2:  # Vertical line
                center_x = x + w // 2
                
                # Check if it's in the center area (where dividers should be)
                if self.image_width * 0.4 < center_x < self.image_width * 0.6:
                    lane_divider_candidates.append(center_x)
        
        # Select the most central divider
        if lane_divider_candidates:
            image_center = self.image_width // 2
            lane_divider_x = min(lane_divider_candidates, key=lambda x: abs(x - image_center))
            self.lane_divider_detected = True
            return lane_divider_x
        
        self.lane_divider_detected = False
        return None
    
    def detect_lines(self, image):
        """
        Detect lines with enhanced parameters for better lane marking detection.
        
        Args:
            image: Masked edge image
            
        Returns:
            lines: Array of detected lines
        """
        lines = cv2.HoughLinesP(
            image,
            rho=1,
            theta=np.pi/180,
            threshold=30,       # Lowered for better detection
            minLineLength=30,   # Shorter for distant markings
            maxLineGap=20       # Larger gap for dashed lines
        )
        
        return lines

    def filter_outliers(self, lines, expected_position=None, side='unknown'):
        """
        Filter out lines that are too far from expected position.
        
        Args:
            lines: List of lines to filter
            expected_position: Expected X position (from previous frame)
            side: 'left' or 'right' for debugging
            
        Returns:
            filtered_lines: Lines that are close to expected position
        """
        if not lines or expected_position is None:
            return lines
            
        filtered_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_center_x = (x1 + x2) / 2
            
            # Check if line is within reasonable distance from expected position
            if abs(line_center_x - expected_position) <= self.outlier_threshold:
                filtered_lines.append(line)
        
        return filtered_lines

    def classify_lanes_with_safety(self, lines, lane_divider_x=None):
        """
        Enhanced lane classification with safety boundaries and divider awareness.
        
        Args:
            lines: Array of detected lines from Hough transform
            lane_divider_x: X-coordinate of detected lane divider
            
        Returns:
            left_lanes: List of left lane lines
            right_lanes: List of right lane lines
            safe_lanes: List of lanes that are safe to follow
        """
        if lines is None:
            return [], [], []
            
        left_lanes = []
        right_lanes = []
        safe_lanes = []
        
        image_center_x = self.image_width // 2
        
        # Enhanced validation parameters
        min_slope_threshold = 0.3
        max_slope_threshold = 3.0
        min_line_length = 25
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Basic line validation
            if x2 - x1 == 0:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            line_center_x = (x1 + x2) / 2
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Validate slope and length
            if abs(slope) < min_slope_threshold or abs(slope) > max_slope_threshold:
                continue
            if line_length < min_line_length:
                continue
                
            # Orientation check - lines should point toward horizon
            if y1 <= y2:  # Wrong direction
                continue
            
            # SAFETY CHECK: Avoid lines too close to sidewalks
            if (line_center_x < self.road_left_boundary or 
                line_center_x > self.road_right_boundary):
                continue
            
            # SAFETY CHECK: If lane divider detected, don't cross it
            if lane_divider_x is not None:
                # If we're on the right side of the road, don't detect lanes on the left side of divider
                if line_center_x < lane_divider_x - 20:  # 20px buffer
                    continue  # This is oncoming traffic lane - ILLEGAL
            
            # Classify based on position
            if line_center_x < image_center_x:
                left_lanes.append(line)
                # Check if it's in our safe driving zone
                if line_center_x > self.lane_center_safe_zone[0] - 50:
                    safe_lanes.append(line)
            else:
                right_lanes.append(line)
                # Check if it's in our safe driving zone
                if line_center_x < self.lane_center_safe_zone[1] + 50:
                    safe_lanes.append(line)
        
        # Filter outliers based on previous frame
        left_lanes = self.filter_outliers(left_lanes, self.prev_left_lane_x, 'left')
        right_lanes = self.filter_outliers(right_lanes, self.prev_right_lane_x, 'right')
        
        return left_lanes, right_lanes, safe_lanes

    def calculate_stable_lane_position(self, lanes, history_deque, prev_position):
        """
        Calculate stable lane position using enhanced temporal filtering and outlier rejection.
        
        Args:
            lanes: List of lane lines
            history_deque: Deque storing historical positions
            prev_position: Previous frame position for validation
            
        Returns:
            stable_position: Smoothed lane position, or None if not confident
        """
        if not lanes:
            return None
            
        # Calculate current frame position
        x_positions = []
        for line in lanes:
            x1, y1, x2, y2 = line[0]
            # Use the bottom point (closer to vehicle) for more accurate steering
            if y1 > y2:  # y1 is lower in image (closer to vehicle)
                x_positions.append(x1)
            else:  # y2 is lower in image
                x_positions.append(x2)
        
        if not x_positions:
            return None
            
        # Current frame average
        current_position = sum(x_positions) / len(x_positions)
        
        # Limit position change to prevent sudden jumps
        if prev_position is not None:
            max_change = 30  # Maximum pixel change between frames
            if abs(current_position - prev_position) > max_change:
                # Limit the change to max_change pixels
                direction = 1 if current_position > prev_position else -1
                current_position = prev_position + (direction * max_change)
        
        # Add to history
        history_deque.append(current_position)
        
        # Calculate smoothed position using exponential weighted average
        # More recent frames have much higher weight
        if len(history_deque) >= 3:  # Need at least 3 frames for stability
            # Exponential weights: recent frames have exponentially more influence
            weights = np.exp(np.linspace(-2, 0, len(history_deque)))
            weights = weights / np.sum(weights)
            stable_position = np.average(list(history_deque), weights=weights)
        else:
            stable_position = current_position
            
        return stable_position

    def calculate_safe_lane_center(self, left_lanes, right_lanes, safe_lanes, lane_divider_x=None):
        """
        Calculate lane center with strict safety enforcement and divider awareness.
        
        Args:
            left_lanes: List of left lane lines
            right_lanes: List of right lane lines
            safe_lanes: List of safe lane lines
            lane_divider_x: X-coordinate of lane divider
            
        Returns:
            lane_center_info: Dictionary with enhanced safety information
        """
        image_center_x = self.image_width // 2
        
        # Calculate stable positions
        stable_left_x = self.calculate_stable_lane_position(
            left_lanes, self.left_lane_history, self.prev_left_lane_x
        )
        stable_right_x = self.calculate_stable_lane_position(
            right_lanes, self.right_lane_history, self.prev_right_lane_x
        )
        
        # Initialize enhanced results
        lane_center_info = {
            'has_left_lane': stable_left_x is not None,
            'has_right_lane': stable_right_x is not None,
            'left_lane_x': stable_left_x,
            'right_lane_x': stable_right_x,
            'lane_center_x': None,
            'steering_error': None,
            'steering_direction': 'UNKNOWN',
            'confidence': 'LOW',
            'safety_status': 'UNKNOWN',
            'lane_divider_x': lane_divider_x,
            'sidewalk_detected': self.sidewalk_detected,
            'illegal_lane_change_risk': False
        }
        
        # SAFETY ENFORCEMENT: Check for illegal lane change risk
        if lane_divider_x is not None and self.prev_lane_center_x is not None:
            # Check if we're getting too close to the divider
            if abs(self.prev_lane_center_x - lane_divider_x) < 40:
                lane_center_info['illegal_lane_change_risk'] = True
                lane_center_info['safety_status'] = 'DANGER'
        
        # Calculate lane center with safety constraints
        current_center = None
        
        if stable_left_x is not None and stable_right_x is not None:
            # Both lanes detected
            lane_width = abs(stable_right_x - stable_left_x)
            
            if 60 <= lane_width <= 180:  # Valid lane width
                current_center = (stable_left_x + stable_right_x) / 2
                lane_center_info['confidence'] = 'HIGH'
                lane_center_info['safety_status'] = 'SAFE'
            else:
                # Invalid width - use safer single lane approach
                if self.prev_lane_center_x is not None:
                    # Stay close to previous position
                    if abs(stable_left_x - self.prev_lane_center_x) < abs(stable_right_x - self.prev_lane_center_x):
                        current_center = stable_left_x + 80
                    else:
                        current_center = stable_right_x - 80
                else:
                    current_center = image_center_x
                    
                lane_center_info['confidence'] = 'MEDIUM'
                lane_center_info['safety_status'] = 'CAUTION'
        
        elif stable_left_x is not None:
            # Only left lane - be very conservative
            if self.prev_lane_center_x is not None:
                estimated_center = stable_left_x + 80
                # Move only 15% toward estimated center
                current_center = self.prev_lane_center_x + 0.15 * (estimated_center - self.prev_lane_center_x)
            else:
                current_center = stable_left_x + 80
                
            lane_center_info['confidence'] = 'LOW'
            lane_center_info['safety_status'] = 'CAUTION'
        
        elif stable_right_x is not None:
            # Only right lane - be very conservative
            if self.prev_lane_center_x is not None:
                estimated_center = stable_right_x - 80
                # Move only 15% toward estimated center
                current_center = self.prev_lane_center_x + 0.15 * (estimated_center - self.prev_lane_center_x)
            else:
                current_center = stable_right_x - 80
                
            lane_center_info['confidence'] = 'LOW'
            lane_center_info['safety_status'] = 'CAUTION'
        
        else:
            # No lanes detected - emergency mode
            if self.prev_lane_center_x is not None:
                current_center = self.prev_lane_center_x  # Hold position
            else:
                current_center = image_center_x
                
            lane_center_info['confidence'] = 'NONE'
            lane_center_info['safety_status'] = 'EMERGENCY'
        
        # CRITICAL SAFETY ENFORCEMENT
        if current_center is not None:
            # Limit sudden changes
            if self.prev_lane_center_x is not None:
                max_change = 15  # Very conservative change limit
                if abs(current_center - self.prev_lane_center_x) > max_change:
                    direction = 1 if current_center > self.prev_lane_center_x else -1
                    current_center = self.prev_lane_center_x + (direction * max_change)
            
            # ABSOLUTE SAFETY BOUNDARIES
            # These boundaries prevent sidewalk approach and illegal lane changes
            absolute_left_limit = self.image_width * 0.30   # 30% - strict left boundary
            absolute_right_limit = self.image_width * 0.70  # 70% - strict right boundary
            
            # If lane divider detected, enforce additional constraint
            if lane_divider_x is not None:
                # Don't go past the divider minus safety margin
                divider_safety_limit = lane_divider_x + 50  # 50px safety margin
                if divider_safety_limit > absolute_left_limit:
                    absolute_left_limit = divider_safety_limit
            
            # Apply absolute constraints
            current_center = np.clip(current_center, absolute_left_limit, absolute_right_limit)
            
            # Update history
            self.lane_center_history.append(current_center)
            
            # Apply temporal smoothing
            if len(self.lane_center_history) >= 5:
                weights = np.exp(np.linspace(-1.5, 0, len(self.lane_center_history)))
                weights = weights / np.sum(weights)
                lane_center_info['lane_center_x'] = np.average(list(self.lane_center_history), weights=weights)
            else:
                lane_center_info['lane_center_x'] = current_center
        
        # Calculate steering with enhanced safety
        if lane_center_info['lane_center_x'] is not None:
            lane_center_info['steering_error'] = lane_center_info['lane_center_x'] - image_center_x
            
            # Safety-based dead zones
            if lane_center_info['safety_status'] == 'DANGER':
                dead_zone = 50  # Large dead zone in danger
            elif lane_center_info['safety_status'] == 'EMERGENCY':
                dead_zone = 60  # Very large dead zone in emergency
            else:
                dead_zone = {
                    'HIGH': 8,
                    'MEDIUM': 15,
                    'LOW': 25,
                    'NONE': 35
                }.get(lane_center_info['confidence'], 35)
            
            if abs(lane_center_info['steering_error']) < dead_zone:
                lane_center_info['steering_direction'] = 'STRAIGHT'
            elif lane_center_info['steering_error'] > 0:
                lane_center_info['steering_direction'] = 'STEER_LEFT'
            else:
                lane_center_info['steering_direction'] = 'STEER_RIGHT'
        
        # Update memory
        self.prev_left_lane_x = stable_left_x
        self.prev_right_lane_x = stable_right_x
        self.prev_lane_center_x = lane_center_info['lane_center_x']
        self.prev_lane_divider_x = lane_divider_x
        
        return lane_center_info

    def process_image(self, image):
        """
        Complete enhanced lane detection pipeline with safety-first approach.
        
        Args:
            image: Raw camera image from CARLA
            
        Returns:
            result_image: Image with comprehensive detection overlay
            lane_info: Dictionary with enhanced lane detection and safety results
        """
        # Step 1: Enhanced preprocessing
        processed = self.preprocess_image(image)
        
        # Step 2: Apply ROI to different detection targets
        masked_lane_edges = self.apply_region_of_interest(processed['lane_edges'])
        
        # Step 3: Detect sidewalks
        sidewalk_boundaries = self.detect_sidewalks(processed['sidewalk_edges'])
        
        # Step 4: Detect lane dividers
        lane_divider_x = self.detect_lane_dividers(processed['yellow_mask'])
        
        # Step 5: Detect lane lines
        lines = self.detect_lines(masked_lane_edges)
        
        # Step 6: Enhanced lane classification with safety
        left_lanes, right_lanes, safe_lanes = self.classify_lanes_with_safety(lines, lane_divider_x)
        
        # Step 7: Calculate safe lane center
        lane_center_info = self.calculate_safe_lane_center(left_lanes, right_lanes, safe_lanes, lane_divider_x)
        
        # Step 8: Create comprehensive result image
        result_image = image.copy()
        
        # Draw detected lanes
        for line in left_lanes:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green for left
            
        for line in right_lanes:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for right
        
        # Draw sidewalk boundaries
        for line in sidewalk_boundaries:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta for sidewalks
        
        # Draw lane divider
        if lane_divider_x is not None:
            cv2.line(result_image, (int(lane_divider_x), 0), (int(lane_divider_x), self.image_height), (0, 255, 255), 3)  # Cyan for divider
        
        # Draw lane center
        if lane_center_info['lane_center_x'] is not None:
            center_x = int(lane_center_info['lane_center_x'])
            color = (0, 255, 0) if lane_center_info['safety_status'] == 'SAFE' else (0, 165, 255)  # Green if safe, orange if not
            cv2.line(result_image, (center_x, 0), (center_x, self.image_height), color, 2)
            cv2.circle(result_image, (center_x, self.image_height - 50), 8, color, -1)
        
        # Draw vehicle center reference
        cv2.line(result_image, (self.image_width // 2, 0), (self.image_width // 2, self.image_height), (255, 255, 255), 1)
        
        # Draw enhanced ROI
        cv2.polylines(result_image, [self.roi_vertices], True, (255, 255, 0), 2)
        
        # Draw safety boundaries
        left_boundary = int(self.image_width * 0.30)
        right_boundary = int(self.image_width * 0.70)
        cv2.line(result_image, (left_boundary, 0), (left_boundary, self.image_height), (128, 128, 128), 1)
        cv2.line(result_image, (right_boundary, 0), (right_boundary, self.image_height), (128, 128, 128), 1)
        
        # Add safety status text
        safety_text = f"Safety: {lane_center_info['safety_status']}"
        cv2.putText(result_image, safety_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if lane_center_info['illegal_lane_change_risk']:
            cv2.putText(result_image, "ILLEGAL LANE CHANGE RISK!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.sidewalk_detected:
            cv2.putText(result_image, "SIDEWALK DETECTED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Prepare comprehensive lane info
        lane_info = {
            'total_lines': len(lines) if lines is not None else 0,
            'left_lanes': len(left_lanes),
            'right_lanes': len(right_lanes),
            'safe_lanes': len(safe_lanes),
            'sidewalk_boundaries': len(sidewalk_boundaries),
            'lane_divider_detected': self.lane_divider_detected,
            'sidewalk_detected': self.sidewalk_detected,
            'edges_image': processed['combined_edges'],
            'masked_edges': masked_lane_edges,
            'lane_center': lane_center_info
        }
        
        return result_image, lane_info 