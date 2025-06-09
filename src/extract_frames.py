import cv2
import os

def extract_frames(video_path, output_dir, prefix, frame_interval=15):
    """Extract every Nth frame from video"""
    cap = cv2.VideoCapture(video_path)
    print(f"Extracting frames from {video_path}")
    frame_count = 0
    saved_count = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        print(f"Frame {frame_count} read")
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            file_name = f"{prefix}_frame_{saved_count:04d}.jpg"
            cv2.imwrite(f"{output_dir}/{file_name}", frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames")
    return saved_count

# Extract frames from all your videos

def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Video paths and output directory
    videos = [
        "birdsView.mp4",
        "frontView.mp4", 
        "horizontalView.mp4",
        "leftView.mp4",
        "rightView.mp4"
    ]
    
    video_dir = os.path.join(curr_dir, "..", "assets", "trainingVideos")
    output_dir = os.path.join(curr_dir, "..", "dataset", "images", "train")
    
    total_frames = 0
    
    for video in videos:
        video_path = os.path.join(video_dir, video)
        if os.path.exists(video_path):
            prefix = video.split(".")[0]
            frames_extracted = extract_frames(video_path, output_dir, prefix, frame_interval=15)
            total_frames += frames_extracted
        else:
            print(f"Warning: {video_path} not found")
    
    print(f"Total frames extracted: {total_frames}")

if __name__ == "__main__":
    main()