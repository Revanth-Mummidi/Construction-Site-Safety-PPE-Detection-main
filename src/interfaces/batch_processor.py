import os
import cv2
import time
from src.core.detector import PPEDetector
from ..config.settings import (
    TARGET_WIDTH, TARGET_HEIGHT,OUTPUT_DIR,SOURCE_DIR,COLORS,COMPLIANCE_THRESHOLD
)


def process_and_save(detector, input_path, output_path):
    """Process an input file and save results"""
    if input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Image processing
        img = cv2.imread(input_path)
        if img is not None:
            frame_data = detector.process_frame(img)
            visualized = visualize_results(frame_data)
            cv2.imwrite(output_path, visualized)
            return True
        return False
    
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        # Video processing
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (TARGET_WIDTH, TARGET_HEIGHT)
        )
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_data = detector.process_frame(frame)
            visualized = visualize_results(frame_data)
            out.write(visualized)
            
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {frame_count} frames ({frame_count/elapsed:.1f} fps)")
        
        cap.release()
        out.release()
        return True
    
    return False

def visualize_results(frame_data):
    """Draw detection results on the frame with enhanced visualization"""
    frame = frame_data['frame'].copy()
    
    # Draw PPE items first
    for item in frame_data['ppe_items']:
        box = [int(x) for x in item['box']]
        color = COLORS.get(item['class'], (0, 255, 0))  # Default to green
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, 
                   f"{item['class']} {item['confidence']:.2f}", 
                   (box[0], box[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 1)
    
    # Draw persons and their scores
    for person in frame_data['persons']:
        box = [int(x) for x in person['box']]
        pid = person['id']
        score = frame_data['scores'].get(pid, 0)
        
        # Determine compliance color
        compliance_color = COLORS['compliant'] if score >= COMPLIANCE_THRESHOLD else COLORS['non_compliant']
        
        # Draw person box
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), COLORS['Person'], 2)
        
        # Draw info box
        info_y = box[1] - 10 if box[1] > 30 else box[3] + 20
        cv2.rectangle(frame, 
                     (box[0] - 1, info_y - 20), 
                     (box[0] + 200, info_y + 60), 
                     compliance_color, -1)
        cv2.putText(frame, f"ID: {pid}", (box[0], info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"PPE: {score}%", (box[0], info_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # List detected PPE items
        if 'ppe_items' in person:
            items_text = ", ".join(person['ppe_items']) if person['ppe_items'] else "None"
            cv2.putText(frame, f"Items: {items_text}", (box[0], info_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def main():
    detector = PPEDetector()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process all files in source directory
    source_files = [f for f in os.listdir(SOURCE_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov'))]
    
    if not source_files:
        print(f"No supported files found in {SOURCE_DIR}")
        return
    
    print(f"Found {len(source_files)} files to process")
    
    for i, filename in enumerate(source_files, 1):
        input_path = os.path.join(SOURCE_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"processed_{filename}")
        
        print(f"Processing {i}/{len(source_files)}: {filename}")
        start_time = time.time()
        
        success = process_and_save(detector, input_path, output_path)
        
        elapsed = time.time() - start_time
        status = "SUCCESS" if success else "FAILED"
        print(f"  {status} in {elapsed:.1f} seconds")
        
        detector.logger.info(f"Processed {filename} - {status}")

if __name__ == "__main__":
    main()