import cv2
import numpy as np
from collections import deque

class LaneDetector:
    """
    Detects lane markings from camera images using computer vision techniques.
    
    This class processes images from CARLA's camera sensor to:
    1. Find lane lines using edge detection
    2. Calculate lane center position with temporal smoothing
    3. Determine steering direction with stability improvements
    """
    
    def __init__(self):
        """Initialize the lane detector with default parameters and temporal filtering."""
        # Image dimensions (should match camera settings)
        self.image_width = 640
        self.image_height = 480
        
        # Region of interest (ROI) - focus on road area
        # These coordinates define a trapezoid shape covering the road
        self.roi_vertices = np.array([
            [0, self.image_height],                    # Bottom left
            [self.image_width // 2 - 50, self.image_height // 2 + 50],  # Top left
            [self.image_width // 2 + 50, self.image_height // 2 + 50],  # Top right
            [self.image_width, self.image_height]      # Bottom right
        ], dtype=np.int32)
        
        # Temporal filtering for stability
        self.history_size = 10  # Number of frames to remember
        self.lane_center_history = deque(maxlen=self.history_size)
        self.left_lane_history = deque(maxlen=self.history_size)
        self.right_lane_history = deque(maxlen=self.history_size)
        
        # Confidence tracking
        self.min_confidence_lines = 3  # Minimum lines needed for confidence
        self.outlier_threshold = 50    # Pixels - lines too far from expected position are outliers
        
        # Previous frame memory for validation
        self.prev_left_lane_x = None
        self.prev_right_lane_x = None
        self.prev_lane_center_x = None
        
        print("Lane detector initialized with temporal filtering")
    
    def preprocess_image(self, image):
        """
        Preprocess the image for lane detection with enhanced noise reduction.
        
        Steps:
        1. Convert to grayscale (edges are easier to detect in grayscale)
        2. Apply stronger Gaussian blur (reduces noise)
        3. Apply morphological operations (further noise reduction)
        4. Apply Canny edge detection (finds edges)
        
        Args:
            image: RGB image from CARLA camera
            
        Returns:
            edges: Binary image with detected edges
        """
        # Step 1: Convert RGB to grayscale
        # Why: Lane lines are usually white/yellow, easier to detect in grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Step 2: Apply stronger Gaussian blur to reduce noise
        # Increased kernel size from (5,5) to (7,7) for better noise reduction
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Step 3: Morphological operations to further reduce noise
        # Create kernel for morphological operations
        kernel = np.ones((3,3), np.uint8)
        # Opening: erosion followed by dilation (removes small noise)
        cleaned = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        
        # Step 4: Canny edge detection with adjusted thresholds
        # Slightly higher thresholds to reduce noise sensitivity
        edges = cv2.Canny(cleaned, 60, 160)
        
        return edges
    
    def apply_region_of_interest(self, image):
        """
        Apply region of interest mask to focus on road area.
        
        Why: We only care about the road in front of the vehicle,
        not the sky, trees, or buildings.
        
        Args:
            image: Edge-detected image
            
        Returns:
            masked_image: Image with only ROI visible
        """
        # Create a mask (black image same size as input)
        mask = np.zeros_like(image)
        
        # Fill the ROI area with white (255)
        cv2.fillPoly(mask, [self.roi_vertices], 255)
        
        # Apply mask: keep only pixels where mask is white
        masked_image = cv2.bitwise_and(image, mask)
        
        return masked_image
    
    def detect_lines(self, image):
        """
        Detect lines in the image using Hough Line Transform with optimized parameters.
        
        Hough Transform finds straight lines in edge-detected images.
        It's perfect for lane detection because lane markings are straight lines.
        
        Args:
            image: Masked edge image
            
        Returns:
            lines: Array of detected lines
        """
        # Optimized Hough Line Transform parameters for reduced noise
        rho = 1              # Distance resolution in pixels
        theta = np.pi/180    # Angular resolution in radians (1 degree)
        threshold = 30       # Reduced from 50 to 30 for smaller image
        min_line_length = 30 # Reduced from 50 to 30 for smaller image
        max_line_gap = 100   # Reduced from 150 to 100 for smaller image
        
        lines = cv2.HoughLinesP(
            image, 
            rho, 
            theta, 
            threshold, 
            np.array([]), 
            minLineLength=min_line_length, 
            maxLineGap=max_line_gap
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
    
    def classify_lanes(self, lines):
        """
        Classify detected lines into left and right lanes using slope analysis and outlier filtering.
        
        Updated Method: Position-based classification with slope filtering and outlier removal
        - Filter lines by reasonable slope range
        - Remove outliers based on previous frame positions
        - Classify based on position relative to image center
        
        Args:
            lines: Array of detected lines from Hough transform
            
        Returns:
            left_lanes: List of lines classified as left lane
            right_lanes: List of lines classified as right lane
        """
        if lines is None:
            return [], []
            
        left_lanes = []
        right_lanes = []
        
        # Image center for position analysis
        image_center_x = self.image_width // 2
        
        # Slope thresholds - adjusted for CARLA's perspective
        min_slope_threshold = 0.05  # Minimum slope magnitude
        max_slope_threshold = 2.0   # Maximum slope magnitude
        
        # First pass: basic classification by position and slope
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope: rise over run
            # Handle vertical lines (divide by zero)
            if x2 - x1 == 0:
                continue  # Skip vertical lines
                
            slope = (y2 - y1) / (x2 - x1)
            line_center_x = (x1 + x2) / 2
            
            # Filter out lines with slopes too close to horizontal or too steep
            if abs(slope) < min_slope_threshold or abs(slope) > max_slope_threshold:
                continue
                
            # Classify based on position
            if line_center_x < image_center_x:
                left_lanes.append(line)
            else:
                right_lanes.append(line)
        
        # Second pass: filter outliers based on previous frame positions
        left_lanes = self.filter_outliers(left_lanes, self.prev_left_lane_x, 'left')
        right_lanes = self.filter_outliers(right_lanes, self.prev_right_lane_x, 'right')
                
        return left_lanes, right_lanes

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

    def calculate_lane_center(self, left_lanes, right_lanes):
        """
        Calculate the center point between left and right lanes with temporal smoothing.
        
        Enhanced Method: Stable Position Calculation with Temporal Filtering
        1. Calculate stable positions for left and right lanes using history
        2. Apply temporal smoothing to reduce fluctuations
        3. Use confidence-based validation
        4. Determine steering direction with hysteresis
        
        Args:
            left_lanes: List of left lane lines
            right_lanes: List of right lane lines
            
        Returns:
            lane_center_info: Dictionary with center calculation results
        """
        # Image center (where vehicle should ideally be)
        image_center_x = self.image_width // 2  # 320 for 640px width
        
        # Calculate stable lane positions
        stable_left_x = self.calculate_stable_lane_position(
            left_lanes, self.left_lane_history, self.prev_left_lane_x
        )
        stable_right_x = self.calculate_stable_lane_position(
            right_lanes, self.right_lane_history, self.prev_right_lane_x
        )
        
        # Initialize results
        lane_center_info = {
            'has_left_lane': stable_left_x is not None,
            'has_right_lane': stable_right_x is not None,
            'left_lane_x': stable_left_x,
            'right_lane_x': stable_right_x,
            'lane_center_x': None,
            'steering_error': None,
            'steering_direction': 'UNKNOWN',
            'confidence': 'LOW'
        }
        
        # Calculate lane center with confidence assessment and boundary validation
        if stable_left_x is not None and stable_right_x is not None:
            # Perfect case: both lanes detected
            current_center = (stable_left_x + stable_right_x) / 2
            lane_center_info['confidence'] = 'HIGH'
            
            # Validate lane width to prevent illegal lane changes
            lane_width = abs(stable_right_x - stable_left_x)
            if lane_width < 60 or lane_width > 200:  # Unrealistic lane width
                # Fallback to single lane detection
                if abs(stable_left_x - self.image_width // 2) < abs(stable_right_x - self.image_width // 2):
                    # Left lane is closer to center, use it
                    estimated_lane_width = 100
                    current_center = stable_left_x + (estimated_lane_width / 2)
                else:
                    # Right lane is closer to center, use it
                    estimated_lane_width = 100
                    current_center = stable_right_x - (estimated_lane_width / 2)
                lane_center_info['confidence'] = 'MEDIUM'
            
        elif stable_left_x is not None:
            # Only left lane: estimate center based on typical lane width
            # Enhanced boundary checking to prevent crossing into oncoming traffic
            estimated_lane_width = 100  # Reduced from 120 to 100 for more conservative estimation
            current_center = stable_left_x + (estimated_lane_width / 2)
            
            # Boundary check: don't go too far right
            if current_center > self.image_width * 0.75:  # Don't go beyond 75% of image width
                current_center = self.image_width * 0.75
            
            lane_center_info['confidence'] = 'MEDIUM'
            
        elif stable_right_x is not None:
            # Only right lane: estimate center based on typical lane width
            # Enhanced boundary checking to prevent crossing into oncoming traffic
            estimated_lane_width = 100  # Reduced from 120 to 100 for more conservative estimation
            current_center = stable_right_x - (estimated_lane_width / 2)
            
            # Boundary check: don't go too far left
            if current_center < self.image_width * 0.25:  # Don't go below 25% of image width
                current_center = self.image_width * 0.25
            
            lane_center_info['confidence'] = 'MEDIUM'
        else:
            # No lanes detected - use previous center if available
            if self.prev_lane_center_x is not None:
                current_center = self.prev_lane_center_x
                lane_center_info['confidence'] = 'LOW'
            else:
                current_center = None
                lane_center_info['confidence'] = 'NONE'
        
        # Apply enhanced temporal smoothing to lane center
        if current_center is not None:
            # Limit sudden changes in lane center
            if self.prev_lane_center_x is not None:
                max_center_change = 25  # Maximum pixel change per frame
                if abs(current_center - self.prev_lane_center_x) > max_center_change:
                    direction = 1 if current_center > self.prev_lane_center_x else -1
                    current_center = self.prev_lane_center_x + (direction * max_center_change)
            
            self.lane_center_history.append(current_center)
            
            # Enhanced smoothing using exponential weighted average
            if len(self.lane_center_history) >= 5:  # Increased from 3 to 5 for better stability
                # Exponential weights with confidence adjustment
                base_weights = np.exp(np.linspace(-2, 0, len(self.lane_center_history)))
                
                # Adjust weights based on confidence
                confidence_factor = {
                    'HIGH': 1.0,
                    'MEDIUM': 0.8,
                    'LOW': 0.6,
                    'NONE': 0.4
                }.get(lane_center_info['confidence'], 0.4)
                
                # Recent frames get more weight with high confidence
                weights = base_weights * confidence_factor
                weights = weights / np.sum(weights)
                
                lane_center_info['lane_center_x'] = np.average(list(self.lane_center_history), weights=weights)
            else:
                # Not enough history, use simple average with current frame bias
                if len(self.lane_center_history) > 1:
                    history_avg = np.mean(list(self.lane_center_history)[:-1])
                    # Blend current frame with history (70% current, 30% history)
                    lane_center_info['lane_center_x'] = 0.7 * current_center + 0.3 * history_avg
                else:
                    lane_center_info['lane_center_x'] = current_center
        
        # Calculate steering error and direction with enhanced hysteresis
        if lane_center_info['lane_center_x'] is not None:
            # Steering error: how far off-center we are
            lane_center_info['steering_error'] = lane_center_info['lane_center_x'] - image_center_x
            
            # Enhanced hysteresis with confidence-based dead zones
            base_dead_zone = {
                'HIGH': 10,    # Smaller dead zone for high confidence
                'MEDIUM': 20,  # Medium dead zone for medium confidence
                'LOW': 30,     # Larger dead zone for low confidence
                'NONE': 40     # Very large dead zone for no confidence
            }.get(lane_center_info['confidence'], 40)
            
            # Additional stability check: if we've been going straight, require larger error to change
            if hasattr(self, 'prev_steering_direction') and self.prev_steering_direction == 'STRAIGHT':
                stability_bonus = 10  # Extra dead zone when previously going straight
            else:
                stability_bonus = 0
            
            dead_zone = base_dead_zone + stability_bonus
            
            if abs(lane_center_info['steering_error']) < dead_zone:
                lane_center_info['steering_direction'] = 'STRAIGHT'
            elif lane_center_info['steering_error'] > 0:
                lane_center_info['steering_direction'] = 'STEER_LEFT'
            else:
                lane_center_info['steering_direction'] = 'STEER_RIGHT'
            
            # Update previous steering direction for next frame
            self.prev_steering_direction = lane_center_info['steering_direction']
        
        # Update previous frame memory
        self.prev_left_lane_x = stable_left_x
        self.prev_right_lane_x = stable_right_x
        self.prev_lane_center_x = lane_center_info['lane_center_x']
        
        return lane_center_info

    def process_image(self, image):
        """
        Complete lane detection pipeline with lane classification and center calculation.
        
        Args:
            image: Raw camera image from CARLA
            
        Returns:
            result_image: Image with lane detection overlay
            lane_info: Dictionary with lane detection results
        """
        # Step 1: Preprocess image
        edges = self.preprocess_image(image)
        
        # Step 2: Apply region of interest
        masked_edges = self.apply_region_of_interest(edges)
        
        # Step 3: Detect lines
        lines = self.detect_lines(masked_edges)
        
        # Step 4: Classify lanes
        left_lanes, right_lanes = self.classify_lanes(lines)
        
        # Step 5: Calculate lane center (NEW!)
        lane_center_info = self.calculate_lane_center(left_lanes, right_lanes)
        
        # Step 6: Create result image (copy of original)
        result_image = image.copy()
        
        # Step 7: Draw classified lanes with different colors
        # Left lanes in RED
        for line in left_lanes:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
        # Right lanes in BLUE
        for line in right_lanes:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        # Step 8: Draw lane center visualization (NEW!)
        if lane_center_info['lane_center_x'] is not None:
            center_x = int(lane_center_info['lane_center_x'])
            # Draw vertical line showing calculated lane center
            cv2.line(result_image, (center_x, 0), (center_x, self.image_height), (0, 255, 255), 2)
            
            # Draw circle at bottom showing lane center point
            cv2.circle(result_image, (center_x, self.image_height - 50), 10, (0, 255, 255), -1)
        
        # Step 9: Draw vehicle center reference
        image_center_x = self.image_width // 2
        cv2.line(result_image, (image_center_x, 0), (image_center_x, self.image_height), (255, 255, 255), 1)
        
        # Step 10: Draw ROI on result image
        cv2.polylines(result_image, [self.roi_vertices], True, (255, 255, 0), 2)
        
        # Step 11: Prepare enhanced lane info
        lane_info = {
            'total_lines': len(lines) if lines is not None else 0,
            'left_lanes': len(left_lanes),
            'right_lanes': len(right_lanes),
            'edges_image': edges,
            'masked_edges': masked_edges,
            'lane_center': lane_center_info  # NEW!
        }
        
        return result_image, lane_info 