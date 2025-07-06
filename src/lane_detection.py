import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from collections import deque
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModernLaneDetector:
    """
    Modern lane detection system using:
    1. YOLOv8 for object detection (vehicles, pedestrians, traffic signs)
    2. Semantic segmentation for precise lane detection
    3. Advanced computer vision techniques used in current research
    """
    
    def __init__(self):
        """Initialize the modern lane detection system."""
        # Image dimensions
        self.image_width = 640
        self.image_height = 480
        
        # Initialize YOLOv8 for object detection
        self.yolo_model = None
        self.segmentation_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self._load_models()
        
        # Detection classes for autonomous driving
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.person_classes = [0]  # person
        self.traffic_classes = [9, 10, 11, 12, 13]  # traffic light, stop sign, etc.
        
        # Lane detection parameters
        self.lane_colors = {
            'ego_lane': [255, 255, 255],      # White lanes
            'adjacent_lane': [255, 255, 0],    # Yellow lanes
            'road_boundary': [0, 255, 0],      # Green boundaries
            'drivable_area': [0, 0, 255]      # Blue drivable area
        }
        
        # Temporal smoothing
        self.detection_history = deque(maxlen=5)
        self.lane_history = deque(maxlen=8)
        
        # Performance metrics
        self.detection_times = deque(maxlen=30)
        self.segmentation_times = deque(maxlen=30)
        
        logger.info(f"Modern lane detector initialized on {self.device}")
    
    def _load_models(self):
        """Load YOLOv8 and segmentation models."""
        try:
            # Load YOLOv8 model for object detection
            self.yolo_model = YOLO('yolov8n.pt')  # Start with nano for speed
            logger.info("YOLOv8 model loaded successfully")
            
            # Load semantic segmentation model for lane detection
            self.segmentation_model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=4,  # ego_lane, adjacent_lane, road_boundary, drivable_area
                activation='sigmoid'
            ).to(self.device)
            
            # Load pretrained weights if available
            try:
                checkpoint = torch.load('models/lane_segmentation.pth', map_location=self.device)
                self.segmentation_model.load_state_dict(checkpoint)
                logger.info("Pretrained segmentation model loaded")
            except FileNotFoundError:
                logger.warning("No pretrained segmentation model found, using random weights")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to basic detection
            self.yolo_model = None
            self.segmentation_model = None
    
    def detect_objects(self, image):
        """
        Detect objects using YOLOv8.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            dict: Detection results with vehicles, pedestrians, traffic signs
        """
        if self.yolo_model is None:
            return {'vehicles': [], 'pedestrians': [], 'traffic_signs': [], 'obstacles': []}
        
        start_time = time.time()
        
        try:
            # Run YOLOv8 detection
            results = self.yolo_model(image, verbose=False)
            
            # Parse results
            detections = {
                'vehicles': [],
                'pedestrians': [],
                'traffic_signs': [],
                'obstacles': []
            }
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Filter by confidence
                        if conf < 0.5:
                            continue
                            
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                        
                        detection = {
                            'bbox': bbox,
                            'confidence': conf,
                            'class': cls,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        
                        # Categorize detections
                        if cls in self.vehicle_classes:
                            detections['vehicles'].append(detection)
                        elif cls in self.person_classes:
                            detections['pedestrians'].append(detection)
                        elif cls in self.traffic_classes:
                            detections['traffic_signs'].append(detection)
                        else:
                            detections['obstacles'].append(detection)
            
            # Record detection time
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return {'vehicles': [], 'pedestrians': [], 'traffic_signs': [], 'obstacles': []}
    
    def segment_lanes(self, image):
        """
        Perform semantic segmentation for lane detection.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            dict: Segmentation results with lane masks
        """
        if self.segmentation_model is None:
            return self._fallback_lane_detection(image)
        
        start_time = time.time()
        
        try:
            # Preprocess image
            input_tensor = self._preprocess_for_segmentation(image)
            
            # Run segmentation
            with torch.no_grad():
                output = self.segmentation_model(input_tensor)
                segmentation_mask = torch.sigmoid(output).cpu().numpy()[0]
            
            # Post-process segmentation results
            lane_info = self._postprocess_segmentation(segmentation_mask, image.shape[:2])
            
            # Record segmentation time
            segmentation_time = time.time() - start_time
            self.segmentation_times.append(segmentation_time)
            
            return lane_info
            
        except Exception as e:
            logger.error(f"Error in lane segmentation: {e}")
            return self._fallback_lane_detection(image)
    
    def _preprocess_for_segmentation(self, image):
        """Preprocess image for segmentation model."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb_image, (256, 256))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return tensor
    
    def _postprocess_segmentation(self, segmentation_mask, original_shape):
        """Post-process segmentation results."""
        h, w = original_shape
        
        # Resize masks back to original size
        ego_lane_mask = cv2.resize(segmentation_mask[0], (w, h))
        adjacent_lane_mask = cv2.resize(segmentation_mask[1], (w, h))
        road_boundary_mask = cv2.resize(segmentation_mask[2], (w, h))
        drivable_area_mask = cv2.resize(segmentation_mask[3], (w, h))
        
        # Apply thresholds
        ego_lane_mask = (ego_lane_mask > 0.5).astype(np.uint8)
        adjacent_lane_mask = (adjacent_lane_mask > 0.5).astype(np.uint8)
        road_boundary_mask = (road_boundary_mask > 0.5).astype(np.uint8)
        drivable_area_mask = (drivable_area_mask > 0.5).astype(np.uint8)
        
        # Extract lane information
        lane_info = {
            'ego_lane_mask': ego_lane_mask,
            'adjacent_lane_mask': adjacent_lane_mask,
            'road_boundary_mask': road_boundary_mask,
            'drivable_area_mask': drivable_area_mask,
            'lane_center': self._calculate_lane_center(ego_lane_mask, drivable_area_mask),
            'lane_confidence': self._calculate_lane_confidence(ego_lane_mask),
            'safe_driving_area': self._calculate_safe_area(drivable_area_mask, road_boundary_mask)
        }
        
        return lane_info
    
    def _calculate_lane_center(self, ego_lane_mask, drivable_area_mask):
        """Calculate the center of the ego lane."""
        if np.sum(ego_lane_mask) == 0:
            # Fallback to drivable area center
            if np.sum(drivable_area_mask) > 0:
                moments = cv2.moments(drivable_area_mask)
                if moments['m00'] != 0:
                    return int(moments['m10'] / moments['m00'])
            return self.image_width // 2
        
        # Find lane center from ego lane mask
        lane_points = np.where(ego_lane_mask > 0)
        if len(lane_points[1]) > 0:
            return int(np.mean(lane_points[1]))
        
        return self.image_width // 2
    
    def _calculate_lane_confidence(self, ego_lane_mask):
        """Calculate confidence in lane detection."""
        total_pixels = ego_lane_mask.shape[0] * ego_lane_mask.shape[1]
        lane_pixels = np.sum(ego_lane_mask)
        
        # Confidence based on lane pixel density
        confidence = min(lane_pixels / (total_pixels * 0.1), 1.0)
        return confidence
    
    def _calculate_safe_area(self, drivable_area_mask, road_boundary_mask):
        """Calculate safe driving area."""
        # Combine drivable area and exclude road boundaries
        safe_area = drivable_area_mask.copy()
        safe_area[road_boundary_mask > 0] = 0
        
        return safe_area
    
    def _fallback_lane_detection(self, image):
        """Fallback lane detection using traditional computer vision."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Create basic lane info
        lane_center = self.image_width // 2
        confidence = 0.3  # Low confidence for fallback
        
        # Create basic masks
        ego_lane_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        drivable_area_mask = np.ones((self.image_height, self.image_width), dtype=np.uint8)
        
        return {
            'ego_lane_mask': ego_lane_mask,
            'adjacent_lane_mask': np.zeros_like(ego_lane_mask),
            'road_boundary_mask': np.zeros_like(ego_lane_mask),
            'drivable_area_mask': drivable_area_mask,
            'lane_center': lane_center,
            'lane_confidence': confidence,
            'safe_driving_area': drivable_area_mask
        }
    
    def process_frame(self, image):
        """
        Process a single frame with modern computer vision techniques.
        
        Args:
            image: Input image from camera
            
        Returns:
            dict: Complete analysis results
        """
        # Object detection
        objects = self.detect_objects(image)
        
        # Lane segmentation
        lanes = self.segment_lanes(image)
        
        # Combine results with temporal smoothing
        results = self._combine_and_smooth_results(objects, lanes)
        
        # Add performance metrics
        results['performance'] = {
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
            'avg_segmentation_time': np.mean(self.segmentation_times) if self.segmentation_times else 0,
            'fps': 1.0 / (results['total_processing_time'] + 1e-6)
        }
        
        return results
    
    def _combine_and_smooth_results(self, objects, lanes):
        """Combine object detection and lane segmentation results with temporal smoothing."""
        # Calculate processing time
        total_time = (np.mean(self.detection_times) if self.detection_times else 0) + \
                    (np.mean(self.segmentation_times) if self.segmentation_times else 0)
        
        # Temporal smoothing for lane center
        current_center = lanes['lane_center']
        if self.lane_history:
            # Weighted average with history
            weights = np.exp(np.linspace(-2, 0, len(self.lane_history)))
            weights = weights / np.sum(weights)
            
            historical_centers = [info['lane_center'] for info in self.lane_history]
            smoothed_center = np.average(historical_centers + [current_center], 
                                       weights=list(weights) + [0.4])
            lanes['lane_center'] = int(smoothed_center)
        
        # Add to history
        self.lane_history.append(lanes.copy())
        
        # Assess safety
        safety_assessment = self._assess_safety(objects, lanes)
        
        results = {
            'objects': objects,
            'lanes': lanes,
            'safety': safety_assessment,
            'lane_center': lanes['lane_center'],
            'lane_confidence': lanes['lane_confidence'],
            'steering_error': lanes['lane_center'] - self.image_width // 2,
            'total_processing_time': total_time
        }
        
        return results
    
    def _assess_safety(self, objects, lanes):
        """Assess safety based on detected objects and lane information."""
        safety_score = 1.0
        warnings = []
        
        # Check for vehicles in ego lane
        ego_lane_mask = lanes['ego_lane_mask']
        for vehicle in objects['vehicles']:
            x1, y1, x2, y2 = vehicle['bbox']
            # Check if vehicle overlaps with ego lane
            vehicle_center_x = (x1 + x2) // 2
            vehicle_center_y = (y1 + y2) // 2
            
            if 0 <= vehicle_center_y < ego_lane_mask.shape[0] and \
               0 <= vehicle_center_x < ego_lane_mask.shape[1]:
                if ego_lane_mask[vehicle_center_y, vehicle_center_x] > 0:
                    safety_score *= 0.5
                    warnings.append(f"Vehicle detected in ego lane (confidence: {vehicle['confidence']:.2f})")
        
        # Check for pedestrians
        for pedestrian in objects['pedestrians']:
            safety_score *= 0.7
            warnings.append(f"Pedestrian detected (confidence: {pedestrian['confidence']:.2f})")
        
        # Check lane confidence
        if lanes['lane_confidence'] < 0.5:
            safety_score *= 0.8
            warnings.append("Low lane detection confidence")
        
        return {
            'safety_score': safety_score,
            'warnings': warnings,
            'emergency_brake': safety_score < 0.3,
            'reduce_speed': safety_score < 0.6
        }
    
    def visualize_results(self, image, results):
        """Visualize detection and segmentation results."""
        vis_image = image.copy()
        
        # Draw object detections
        self._draw_object_detections(vis_image, results['objects'])
        
        # Draw lane segmentation
        self._draw_lane_segmentation(vis_image, results['lanes'])
        
        # Draw safety information
        self._draw_safety_info(vis_image, results['safety'])
        
        # Draw performance metrics
        self._draw_performance_metrics(vis_image, results.get('performance', {}))
        
        return vis_image
    
    def _draw_object_detections(self, image, objects):
        """Draw object detection results."""
        # Draw vehicles (red)
        for vehicle in objects['vehicles']:
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f"Vehicle {vehicle['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw pedestrians (orange)
        for pedestrian in objects['pedestrians']:
            x1, y1, x2, y2 = pedestrian['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(image, f"Person {pedestrian['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # Draw traffic signs (green)
        for sign in objects['traffic_signs']:
            x1, y1, x2, y2 = sign['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Sign {sign['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _draw_lane_segmentation(self, image, lanes):
        """Draw lane segmentation results."""
        # Create colored overlay
        overlay = np.zeros_like(image)
        
        # Draw ego lane (white)
        overlay[lanes['ego_lane_mask'] > 0] = [255, 255, 255]
        
        # Draw adjacent lanes (yellow)
        overlay[lanes['adjacent_lane_mask'] > 0] = [0, 255, 255]
        
        # Draw road boundaries (green)
        overlay[lanes['road_boundary_mask'] > 0] = [0, 255, 0]
        
        # Draw drivable area (blue, semi-transparent)
        overlay[lanes['drivable_area_mask'] > 0] = [255, 0, 0]
        
        # Blend with original image
        cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
        
        # Draw lane center line
        center_x = lanes['lane_center']
        cv2.line(image, (center_x, 0), (center_x, image.shape[0]), (0, 255, 255), 2)
    
    def _draw_safety_info(self, image, safety):
        """Draw safety information."""
        # Safety score
        color = (0, 255, 0) if safety['safety_score'] > 0.7 else \
                (0, 255, 255) if safety['safety_score'] > 0.4 else (0, 0, 255)
        
        cv2.putText(image, f"Safety: {safety['safety_score']:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Warnings
        y_offset = 60
        for warning in safety['warnings']:
            cv2.putText(image, warning, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 20
    
    def _draw_performance_metrics(self, image, performance):
        """Draw performance metrics."""
        if not performance:
            return
            
        y_pos = image.shape[0] - 60
        cv2.putText(image, f"FPS: {performance.get('fps', 0):.1f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(image, f"Det: {performance.get('avg_detection_time', 0)*1000:.1f}ms", 
                   (10, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(image, f"Seg: {performance.get('avg_segmentation_time', 0)*1000:.1f}ms", 
                   (10, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def get_status(self):
        """Get current system status."""
        return {
            'models_loaded': self.yolo_model is not None and self.segmentation_model is not None,
            'device': str(self.device),
            'yolo_available': self.yolo_model is not None,
            'segmentation_available': self.segmentation_model is not None,
            'avg_fps': 1.0 / (np.mean(self.detection_times) + np.mean(self.segmentation_times) + 1e-6) if self.detection_times and self.segmentation_times else 0
        } 