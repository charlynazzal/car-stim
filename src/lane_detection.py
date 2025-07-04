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
        Preprocess the image for lane detection.
        
        Steps:
        1. Convert to grayscale (edges are easier to detect in grayscale)
        2. Apply Gaussian blur (reduces noise)
        3. Apply Canny edge detection (finds edges)
        
        Args:
            image: RGB image from CARLA camera
            
        Returns:
            edges: Binary image with detected edges
        """
        # Step 1: Convert RGB to grayscale
        # Why: Lane lines are usually white/yellow, easier to detect in grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Step 2: Apply Gaussian blur to reduce noise
        # Why: Removes small details that aren't lane lines
        # Kernel size (5,5) means 5x5 pixel averaging
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 3: Canny edge detection
        # Why: Finds rapid changes in intensity (edges)
        # Parameters: low_threshold=50, high_threshold=150
        edges = cv2.Canny(blurred, 50, 150)
        
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
        Detect lines in the image using Hough Line Transform.
        
        Hough Transform finds straight lines in edge-detected images.
        It's perfect for lane detection because lane markings are straight lines.
        
        Args:
            image: Masked edge image
            
        Returns:
            lines: Array of detected lines
        """
        # Hough Line Transform parameters
        rho = 1              # Distance resolution in pixels
        theta = np.pi/180    # Angular resolution in radians (1 degree)
        threshold = 50       # Minimum number of votes (intersections in Hough space)
        min_line_length = 50 # Minimum line length
        max_line_gap = 150   # Maximum gap between line segments
        
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
        Calculate stable lane position using confidence-based filtering and temporal smoothing.
        
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
        
        # Add to history
        history_deque.append(current_position)
        
        # Calculate smoothed position using weighted average
        # More recent frames have higher weight
        if len(history_deque) >= 3:  # Need at least 3 frames for stability
            weights = np.linspace(0.1, 1.0, len(history_deque))  # Recent frames weighted higher
            weighted_positions = np.array(history_deque) * weights
            stable_position = np.sum(weighted_positions) / np.sum(weights)
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
        
        # Calculate lane center with confidence assessment
        if stable_left_x is not None and stable_right_x is not None:
            # Perfect case: both lanes detected
            current_center = (stable_left_x + stable_right_x) / 2
            lane_center_info['confidence'] = 'HIGH'
            
        elif stable_left_x is not None:
            # Only left lane: estimate center based on typical lane width
            estimated_lane_width = 200  # pixels
            current_center = stable_left_x + (estimated_lane_width / 2)
            lane_center_info['confidence'] = 'MEDIUM'
            
        elif stable_right_x is not None:
            # Only right lane: estimate center based on typical lane width
            estimated_lane_width = 200  # pixels
            current_center = stable_right_x - (estimated_lane_width / 2)
            lane_center_info['confidence'] = 'MEDIUM'
        else:
            # No lanes detected - use previous center if available
            if self.prev_lane_center_x is not None:
                current_center = self.prev_lane_center_x
                lane_center_info['confidence'] = 'LOW'
            else:
                current_center = None
                lane_center_info['confidence'] = 'NONE'
        
        # Apply temporal smoothing to lane center
        if current_center is not None:
            self.lane_center_history.append(current_center)
            
            # Smooth lane center using weighted average
            if len(self.lane_center_history) >= 3:
                weights = np.linspace(0.1, 1.0, len(self.lane_center_history))
                weighted_centers = np.array(self.lane_center_history) * weights
                lane_center_info['lane_center_x'] = np.sum(weighted_centers) / np.sum(weights)
            else:
                lane_center_info['lane_center_x'] = current_center
        
        # Calculate steering error and direction with hysteresis
        if lane_center_info['lane_center_x'] is not None:
            # Steering error: how far off-center we are
            lane_center_info['steering_error'] = lane_center_info['lane_center_x'] - image_center_x
            
            # Determine steering direction with hysteresis (larger dead zone for stability)
            dead_zone = 15 if lane_center_info['confidence'] == 'HIGH' else 25
            
            if abs(lane_center_info['steering_error']) < dead_zone:
                lane_center_info['steering_direction'] = 'STRAIGHT'
            elif lane_center_info['steering_error'] > 0:
                lane_center_info['steering_direction'] = 'STEER_LEFT'
            else:
                lane_center_info['steering_direction'] = 'STEER_RIGHT'
        
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