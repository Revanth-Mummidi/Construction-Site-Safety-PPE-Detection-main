import cv2
import os
import time
import logging
from collections import defaultdict
from ultralytics import YOLO
from ..config.settings import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, IMG_SIZE,
    TARGET_WIDTH, TARGET_HEIGHT, IOU_THRESHOLD,
    PPE_WEIGHTS , LOG_DIR
)

class PPEDetector:
    def __init__(self):
         # Add these debug statements right at the start of the __init__ method
        print(f"Attempting to load model from: {MODEL_PATH}")
        print(f"File exists: {os.path.exists(MODEL_PATH)}")
        
        # Add the validation check before loading the model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at: {MODEL_PATH}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Please ensure: \n"
                f"1. The model file exists at the specified path\n"
                f"2. You have proper read permissions\n"
                f"3. The path is correctly specified in src/config/settings.py"
            )
        self.model = YOLO(MODEL_PATH)
        self.logger = self.setup_logging()
        self.id_counter = 0
        self.tracked_persons = {}
        
    def setup_logging(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(LOG_DIR, 'ppe_detection.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger()
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes"""
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union
    
    def assign_ppe_to_persons(self, persons, ppe_items):
        assigned_ppe = defaultdict(list)
        
        # Sort PPE items by size (largest first) to prioritize vests/helmets
        ppe_items_sorted = sorted(
            ppe_items,
            key=lambda x: (x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1]),
            reverse=True
        )
        
        for ppe in ppe_items_sorted:
            best_iou = 0
            best_person = None
            
            for person in persons:
                current_iou = self.calculate_iou(person['box'], ppe['box'])
                if current_iou > best_iou and current_iou >= IOU_THRESHOLD:
                    best_iou = current_iou
                    best_person = person
            
            if best_person is not None:
                assigned_ppe[best_person['id']].append(ppe)
        
        return assigned_ppe
    
    def calculate_ppe_scores(self, persons, assigned_ppe):
        scores = {}
        for person in persons:
            pid = person['id']
            score = 0
            
            # Calculate base score from detected items
            if pid in assigned_ppe:
                for ppe in assigned_ppe[pid]:
                    score += PPE_WEIGHTS.get(ppe['class'], 0)
            
            # Enforce 100% cap and minimum 0%
            scores[pid] = max(0, min(100, score))
            
            # Debug output
            print(f"Person {pid} PPE Items: {[item['class'] for item in assigned_ppe.get(pid, [])]}")
            print(f"Raw Score: {score} -> Final Score: {scores[pid]}%")
        
        return scores

    def track_persons(self, current_persons):
        """Track persons across frames with ID persistence"""
        updated_persons = []
        
        for person in current_persons:
            # Simple tracking based on box center proximity
            cx = (person['box'][0] + person['box'][2]) / 2
            cy = (person['box'][1] + person['box'][3]) / 2
            
            matched_id = None
            min_distance = float('inf')
            
            for pid, data in self.tracked_persons.items():
                prev_cx, prev_cy = data['center']
                distance = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
                
                if distance < 50 and distance < min_distance:  # 50 pixel threshold
                    min_distance = distance
                    matched_id = pid
            
            if matched_id is not None:
                person['id'] = matched_id
                # Update tracker with new position
                self.tracked_persons[matched_id] = {
                    'center': (cx, cy),
                    'timestamp': time.time()
                }
            else:
                # New person
                self.id_counter += 1
                person['id'] = self.id_counter
                self.tracked_persons[self.id_counter] = {
                    'center': (cx, cy),
                    'timestamp': time.time()
                }
            
            updated_persons.append(person)
        
        # Clean up old tracks (people who left the frame)
        current_time = time.time()
        self.tracked_persons = {
            pid: data for pid, data in self.tracked_persons.items()
            if current_time - data['timestamp'] < 2.0  # 2 second timeout
        }
        
        return updated_persons
    
    def process_frame(self, frame):
        """Process a single frame and return results"""
        try:
            # Resize to target resolution
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            
            # Run YOLO inference
            results = self.model(frame, imgsz=IMG_SIZE, conf=CONFIDENCE_THRESHOLD)
            
            # Convert detections to readable format
            detections = []
            for result in results:
                for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    detections.append({
                        'class': self.model.names[int(cls)],
                        'box': box.tolist(),
                        'confidence': float(conf)
                    })
            
            # Separate persons and PPE items
            persons = [d for d in detections if d['class'] == 'Person']
            ppe_items = [d for d in detections if d['class'] in PPE_WEIGHTS]
            
            # Track persons across frames
            tracked_persons = self.track_persons(persons)
            
            # Assign PPE items to persons
            assigned_ppe = self.assign_ppe_to_persons(tracked_persons, ppe_items)
            
            # Calculate PPE scores
            scores = self.calculate_ppe_scores(tracked_persons, assigned_ppe)
            
            return {
                'frame': frame,
                'persons': tracked_persons,
                'ppe_items': ppe_items,
                'scores': scores
            }
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            raise