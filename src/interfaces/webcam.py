import cv2
import time
from src.core.detector import PPEDetector
from ..config.settings import (
    TARGET_WIDTH, TARGET_HEIGHT, TARGET_FPS,
    COMPLIANCE_THRESHOLD, COLORS
)

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
    cap = cv2.VideoCapture(0)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    frame_time = 1.0 / TARGET_FPS
    
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_data = detector.process_frame(frame)
            visualized = visualize_results(frame_data)
            
            # Show results
            cv2.imshow("PPE Compliance Monitor", visualized)
            
            # Control frame rate
            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            
            if cv2.waitKey(1) == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()