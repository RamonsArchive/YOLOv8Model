from ultralytics import YOLO
import os
import pandas as pd
import cv2
import numpy as np

def run_best_pt(best_pt_path, video_path, output_dir):
    model = YOLO(best_pt_path)
    
    # Open video and check original frame
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if ret:
        print(f"Original frame shape: {first_frame.shape}")
        # Rotate frame 90 degrees clockwise to match training data
        rotated_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
        print(f"Rotated frame shape: {rotated_frame.shape}")
        cv2.imwrite(f"{output_dir}/debug_original_frame.jpg", first_frame)
        cv2.imwrite(f"{output_dir}/debug_rotated_frame.jpg", rotated_frame)
    cap.release()
    
    # Create a temporary rotated video
    temp_video_path = f"{output_dir}/temp_rotated_video.mp4"
    rotate_video(video_path, temp_video_path)
    
    # Run detection on rotated video
    results = model.track(
        source=temp_video_path,
        save=True,
        project=output_dir,
        name='detection',
        conf=0.3,
        imgsz=640,  # Single value should work better
        stream=True
    )
    
    # Save detection data to CSV
    detections = []
    frame_num = 0
    
    for result in results:
        frame_num += 1
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                bbox = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                detections.append({
                    'frame_number': frame_num,
                    'object_class': model.names[cls],
                    'confidence_score': float(conf),
                    'x1': float(bbox[0]),
                    'y1': float(bbox[1]),
                    'x2': float(bbox[2]),
                    'y2': float(bbox[3])
                })
    
    # Clean up temporary video
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    
    # Save to CSV
    if detections:
        df = pd.DataFrame(detections)
        csv_path = f'{output_dir}/detection_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(detections)} detections to {csv_path}")
        
        # Print summary
        print("\nDetection Summary:")
        print(df['object_class'].value_counts())
        return csv_path
    else:
        return None

def rotate_video(input_path, output_path):
    """Rotate video 90 degrees clockwise"""
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get first frame to determine output dimensions
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return
    
    # Rotate first frame to get new dimensions
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    height, width = rotated_frame.shape[:2]
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Reset video capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"Rotating video: {frame_count} frames at {fps} FPS")
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Rotate frame 90 degrees clockwise
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        out.write(rotated_frame)
        
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"Processed {frame_num}/{frame_count} frames")
    
    cap.release()
    out.release()
    print(f"Rotated video saved to: {output_path}")

def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    best_pt_path = os.path.join(curr_dir, "..", "runs/train/desk_objects/weights/best.pt")
    video_path = os.path.join(curr_dir, "..", "assets/testVideo/deskVideo.mp4")
    output_dir = os.path.join(curr_dir, "..", "runs/train/desk_objects/")
    
    run_best_pt(best_pt_path, video_path, output_dir)
    print("Running best.pt")

if __name__ == "__main__":
    main()