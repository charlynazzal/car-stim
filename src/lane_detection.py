import cv2
import numpy as np
from collections import deque

class LaneDetector:
    """
    Robust lane detection system for autonomous driving.
    
    This system implements a multi-stage approach:
    1. Road segmentation to identify drivable areas
    2. Lane detection within road boundaries only
    3. Path planning within detected lanes
    4. Basic obstacle awareness
    """
    
    def __init__(self):
        """Initialize the robust lane detector."""
        # Image dimensions
        self.image_width = 640
        self.image_height = 480
        
        # Define a more focused ROI that covers the road ahead
        self.roi_vertices = np.array([
            [0, self.image_height - 80],                    # Bottom left
            [self.image_width // 2 - 150, self.image_height // 2 + 20],  # Top left
            [self.image_width // 2 + 150, self.image_height // 2 + 20],  # Top right  
            [self.image_width - 1, self.image_height - 80]  # Bottom right
        ], dtype=np.int32)
        
        # Temporal filtering
        self.history_size = 10
        self.lane_center_history = deque(maxlen=self.history_size)
        self.road_mask_history = deque(maxlen=5)  # For road segmentation stability
        
        # Lane detection parameters
        self.min_lane_width = 80   # Minimum pixels between lane lines
        self.max_lane_width = 200  # Maximum pixels between lane lines
        self.outlier_threshold = 40  # Reduced for more stable detection
        
        # Previous frame memory
        self.prev_lane_center_x = None
        self.prev_left_lane_x = None
        self.prev_right_lane_x = None
        
        # Road segmentation parameters
        self.road_color_lower = np.array([0, 0, 50])    # Dark road surface
        self.road_color_upper = np.array([180, 50, 120]) # Light road surface
        
        print("Robust lane detector initialized")
        print(f"ROI covers road area from y={self.image_height//2 + 20} to y={self.image_height-80}")
    
    def segment_road(self, image):
        """
        Segment the road area from the image using color and texture analysis.
        
        This is the foundation - we only look for lanes within the road area.
        
        Args:
            image: RGB image from camera
            
        Returns:
            road_mask: Binary mask where road pixels are white
        """
        # Convert to different color spaces for robust road detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Color-based road detection
        # Roads are typically gray/dark with low saturation
        road_mask_color = cv2.inRange(hsv, self.road_color_lower, self.road_color_upper)
        
        # Method 2: Texture-based road detection
        # Roads have relatively uniform texture
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        texture_diff = cv2.absdiff(gray, blurred)
        _, road_mask_texture = cv2.threshold(texture_diff, 25, 255, cv2.THRESH_BINARY_INV)
        
        # Method 3: Gradient-based detection
        # Roads have fewer sharp edges than buildings/sidewalks
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        road_mask_gradient = cv2.bitwise_not(edges_dilated)
        
        # Combine all methods
        road_mask = cv2.bitwise_and(road_mask_color, road_mask_texture)
        road_mask = cv2.bitwise_and(road_mask, road_mask_gradient)
        
        # Clean up the mask
        kernel = np.ones((7, 7), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply ROI to focus on the road ahead
        roi_mask = np.zeros_like(road_mask)
        cv2.fillPoly(roi_mask, [self.roi_vertices], 255)
        road_mask = cv2.bitwise_and(road_mask, roi_mask)
        
        # Temporal smoothing for stability
        self.road_mask_history.append(road_mask)
        if len(self.road_mask_history) >= 3:
            # Use majority voting across recent frames
            stacked = np.stack(list(self.road_mask_history), axis=0)
            road_mask = np.where(np.sum(stacked > 0, axis=0) >= 2, 255, 0).astype(np.uint8)
        
        return road_mask
    
    def detect_lane_lines_in_road(self, image, road_mask):
        """
        Detect lane lines only within the segmented road area.
        
        Args:
            image: RGB image
            road_mask: Binary mask of road area
            
        Returns:
            left_lanes, right_lanes: Lists of detected lane lines
        """
        # Convert to HSV for better lane marking detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Detect white lane markings
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Detect yellow lane markings
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Combine lane markings
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Only look for lanes within the road area
        lane_mask = cv2.bitwise_and(lane_mask, road_mask)
        
        # Apply edge detection to find lane edges
        edges = cv2.Canny(lane_mask, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=40,
            maxLineGap=25
        )
        
        if lines is None:
            return [], []
        
        # Classify lines into left and right lanes
        left_lanes = []
        right_lanes = []
        image_center_x = self.image_width // 2
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line properties
            if x2 - x1 == 0:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            line_center_x = (x1 + x2) / 2
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Filter by slope (lanes should have reasonable slope)
            if abs(slope) < 0.3 or abs(slope) > 3.0:
                continue
                
            # Filter by length
            if line_length < 30:
                continue
            
            # Check if line points toward horizon (y1 > y2 for valid lanes)
            if y1 <= y2:
                continue
            
            # Classify based on position and slope
            if line_center_x < image_center_x and slope < 0:
                # Left lane (negative slope)
                left_lanes.append(line)
            elif line_center_x > image_center_x and slope > 0:
                # Right lane (positive slope)  
                right_lanes.append(line)
        
        return left_lanes, right_lanes
    
    def calculate_lane_center_robust(self, left_lanes, right_lanes, road_mask):
        """
        Calculate lane center using multiple methods for robustness.
        
        Args:
            left_lanes: List of left lane lines
            right_lanes: List of right lane lines
            road_mask: Binary mask of road area
            
        Returns:
            lane_center_info: Dictionary with lane center calculation
        """
        image_center_x = self.image_width // 2
        
        # Method 1: Use detected lane lines
        left_x = None
        right_x = None
        
        if left_lanes:
            # Average all left lane positions at bottom of image
            left_positions = []
            for line in left_lanes:
                x1, y1, x2, y2 = line[0]
                # Extrapolate to bottom of image
                if y1 != y2:
                    bottom_x = x1 + (x2 - x1) * (self.image_height - 80 - y1) / (y2 - y1)
                    left_positions.append(bottom_x)
            if left_positions:
                left_x = np.mean(left_positions)
        
        if right_lanes:
            # Average all right lane positions at bottom of image
            right_positions = []
            for line in right_lanes:
                x1, y1, x2, y2 = line[0]
                # Extrapolate to bottom of image
                if y1 != y2:
                    bottom_x = x1 + (x2 - x1) * (self.image_height - 80 - y1) / (y2 - y1)
                    right_positions.append(bottom_x)
            if right_positions:
                right_x = np.mean(right_positions)
        
        # Method 2: Use road mask center if no lanes detected
        road_center_x = None
        if left_x is None and right_x is None:
            # Find the center of the road mask at the bottom
            bottom_row = road_mask[self.image_height - 100:self.image_height - 80, :]
            road_pixels = np.where(bottom_row > 0)
            if len(road_pixels[1]) > 0:
                road_left = np.min(road_pixels[1])
                road_right = np.max(road_pixels[1])
                road_center_x = (road_left + road_right) / 2
        
        # Calculate lane center
        lane_center_x = None
        confidence = 'NONE'
        
        if left_x is not None and right_x is not None:
            # Both lanes detected
            lane_width = abs(right_x - left_x)
            if self.min_lane_width <= lane_width <= self.max_lane_width:
                lane_center_x = (left_x + right_x) / 2
                confidence = 'HIGH'
            else:
                # Invalid lane width, use single lane approach
                if self.prev_lane_center_x is not None:
                    if abs(left_x - self.prev_lane_center_x) < abs(right_x - self.prev_lane_center_x):
                        lane_center_x = left_x + 90  # Assume 180px lane width
                    else:
                        lane_center_x = right_x - 90
                else:
                    lane_center_x = (left_x + right_x) / 2
                confidence = 'MEDIUM'
                
        elif left_x is not None:
            # Only left lane detected
            lane_center_x = left_x + 90  # Assume we're 90px from left edge
            confidence = 'LOW'
            
        elif right_x is not None:
            # Only right lane detected
            lane_center_x = right_x - 90  # Assume we're 90px from right edge
            confidence = 'LOW'
            
        elif road_center_x is not None:
            # No lanes, but road detected - follow road center
            lane_center_x = road_center_x
            confidence = 'LOW'
            
        else:
            # Nothing detected - use previous position or image center
            if self.prev_lane_center_x is not None:
                lane_center_x = self.prev_lane_center_x
            else:
                lane_center_x = image_center_x
            confidence = 'NONE'
        
        # Apply temporal smoothing and safety constraints
        if lane_center_x is not None:
            # Limit sudden changes
            if self.prev_lane_center_x is not None:
                max_change = 30
                if abs(lane_center_x - self.prev_lane_center_x) > max_change:
                    direction = 1 if lane_center_x > self.prev_lane_center_x else -1
                    lane_center_x = self.prev_lane_center_x + (direction * max_change)
            
            # Keep within safe boundaries (20% to 80% of image width)
            safe_left = self.image_width * 0.20
            safe_right = self.image_width * 0.80
            lane_center_x = np.clip(lane_center_x, safe_left, safe_right)
            
            # Add to history for smoothing
            self.lane_center_history.append(lane_center_x)
            
            # Apply temporal smoothing
            if len(self.lane_center_history) >= 5:
                weights = np.exp(np.linspace(-1, 0, len(self.lane_center_history)))
                weights = weights / np.sum(weights)
                lane_center_x = np.average(list(self.lane_center_history), weights=weights)
        
        # Calculate steering error
        steering_error = None
        steering_direction = 'UNKNOWN'
        
        if lane_center_x is not None:
            steering_error = lane_center_x - image_center_x
            
            # Adaptive dead zone based on confidence
            dead_zone = {
                'HIGH': 15,
                'MEDIUM': 25,
                'LOW': 35,
                'NONE': 50
            }.get(confidence, 50)
            
            if abs(steering_error) < dead_zone:
                steering_direction = 'STRAIGHT'
            elif steering_error > 0:
                steering_direction = 'STEER_LEFT'
            else:
                steering_direction = 'STEER_RIGHT'
        
        # Update memory
        self.prev_lane_center_x = lane_center_x
        self.prev_left_lane_x = left_x
        self.prev_right_lane_x = right_x
        
        return {
            'has_left_lane': left_x is not None,
            'has_right_lane': right_x is not None,
            'left_lane_x': left_x,
            'right_lane_x': right_x,
            'lane_center_x': lane_center_x,
            'steering_error': steering_error,
            'steering_direction': steering_direction,
            'confidence': confidence,
            'safety_status': 'SAFE' if confidence in ['HIGH', 'MEDIUM'] else 'CAUTION'
        }
    
    def detect_obstacles_basic(self, image, road_mask):
        """
        Basic obstacle detection within the road area.
        
        This is a simple implementation - for full functionality,
        you'd want to use CARLA's LiDAR or depth sensors.
        
        Args:
            image: RGB image
            road_mask: Binary mask of road area
            
        Returns:
            obstacles: List of detected obstacle bounding boxes
        """
        # Convert to HSV for better object detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Detect non-road objects (cars, pedestrians, etc.)
        # Cars are typically darker than the road
        car_lower = np.array([0, 0, 0])
        car_upper = np.array([180, 255, 80])
        car_mask = cv2.inRange(hsv, car_lower, car_upper)
        
        # Only look for obstacles within the road area
        obstacle_mask = cv2.bitwise_and(car_mask, road_mask)
        
        # Find contours (potential obstacles)
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                # Only consider obstacles in the lower half of the image (closer to vehicle)
                if y > self.image_height // 2:
                    obstacles.append((x, y, w, h))
        
        return obstacles
    
    def process_image(self, image):
        """
        Complete robust lane detection pipeline.
        
        Args:
            image: Raw camera image from CARLA
            
        Returns:
            result_image: Image with detection overlays
            lane_info: Dictionary with lane detection results
        """
        # Step 1: Segment the road area
        road_mask = self.segment_road(image)
        
        # Step 2: Detect lane lines within the road area
        left_lanes, right_lanes = self.detect_lane_lines_in_road(image, road_mask)
        
        # Step 3: Calculate robust lane center
        lane_center_info = self.calculate_lane_center_robust(left_lanes, right_lanes, road_mask)
        
        # Step 4: Basic obstacle detection
        obstacles = self.detect_obstacles_basic(image, road_mask)
        
        # Step 5: Create result image with overlays
        result_image = image.copy()
        
        # Draw road mask (semi-transparent green overlay)
        road_overlay = np.zeros_like(image)
        road_overlay[road_mask > 0] = [0, 255, 0]  # Green for road
        result_image = cv2.addWeighted(result_image, 0.8, road_overlay, 0.2, 0)
        
        # Draw detected lane lines
        for line in left_lanes:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow for left
            
        for line in right_lanes:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Magenta for right
        
        # Draw lane center
        if lane_center_info['lane_center_x'] is not None:
            center_x = int(lane_center_info['lane_center_x'])
            color = (0, 255, 0) if lane_center_info['confidence'] in ['HIGH', 'MEDIUM'] else (0, 165, 255)
            cv2.line(result_image, (center_x, 0), (center_x, self.image_height), color, 2)
            cv2.circle(result_image, (center_x, self.image_height - 50), 8, color, -1)
        
        # Draw vehicle center reference
        cv2.line(result_image, (self.image_width // 2, 0), (self.image_width // 2, self.image_height), (255, 255, 255), 1)
        
        # Draw ROI
        cv2.polylines(result_image, [self.roi_vertices], True, (255, 255, 0), 2)
        
        # Draw obstacles
        for (x, y, w, h) in obstacles:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result_image, "OBSTACLE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add status text
        status_text = f"Confidence: {lane_center_info['confidence']} | Status: {lane_center_info['safety_status']}"
        cv2.putText(result_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if obstacles:
            cv2.putText(result_image, f"OBSTACLES DETECTED: {len(obstacles)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Prepare lane info
        lane_info = {
            'total_lines': len(left_lanes) + len(right_lanes),
            'left_lanes': len(left_lanes),
            'right_lanes': len(right_lanes),
            'obstacles_detected': len(obstacles),
            'road_area_detected': np.sum(road_mask > 0) > 1000,  # Minimum road area
            'lane_center': lane_center_info
        }
        
        return result_image, lane_info 