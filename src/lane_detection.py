import cv2
import numpy as np

class LaneDetector:
    """
    Detects lane markings from camera images using computer vision techniques.
    
    This class processes images from CARLA's camera sensor to:
    1. Find lane lines using edge detection
    2. Calculate lane center position
    3. Determine steering direction
    """
    
    def __init__(self):
        """Initialize the lane detector with default parameters."""
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
        
        print("Lane detector initialized")
    
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
    
    def process_image(self, image):
        """
        Complete lane detection pipeline.
        
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
        
        # Step 4: Create result image (copy of original)
        result_image = image.copy()
        
        # Step 5: Draw detected lines on result image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Step 6: Draw ROI on result image
        cv2.polylines(result_image, [self.roi_vertices], True, (255, 0, 0), 2)
        
        # Prepare lane info (we'll expand this later)
        lane_info = {
            'lines_detected': len(lines) if lines is not None else 0,
            'edges_image': edges,
            'masked_edges': masked_edges
        }
        
        return result_image, lane_info 