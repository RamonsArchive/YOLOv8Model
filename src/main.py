#!/usr/bin/env python3
"""
Complete YOLO Pipeline for Assignment
This script handles everything needed for your take-home task
"""

import os
import json
import shutil
import random
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import yaml

def convert_json_to_yolo(json_path, img_width, img_height, class_mapping):
    """
    Convert LabelMe JSON annotation to YOLO format
    Handles both rectangle and polygon shapes
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    yolo_annotations = []
    
    # Process LabelMe format
    if 'shapes' in data:
        for shape in data['shapes']:
            label = shape['label']
            
            # Map label variations to standard names
            label_mapping = {
                'water_bottle': 'water_bottle',
                'mouse': 'mouse',
                'airpod_case': 'airpod_case',
                'mechanical_pencil': 'mechanical_pencil',
                'keys': 'keys',
                'guitar_pick': 'guitar_pick',
                'keyboard': 'keyboard',
                'monitor': 'monitor',
                'laptop': 'laptop',
            }
            
            # Apply label mapping if needed
            if label in label_mapping:
                label = label_mapping[label]
            
            if label in class_mapping:
                class_id = class_mapping[label]
                points = shape['points']
                
                if shape['shape_type'] == 'rectangle':
                    # Rectangle: two points [top-left, bottom-right]
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    
                elif shape['shape_type'] == 'polygon':
                    # Polygon: convert to bounding box
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                
                else:
                    continue  # Skip unsupported shapes
                
                # Convert to YOLO format (normalized center coordinates + width/height)
                center_x = (x1 + x2) / 2 / img_width
                center_y = (y1 + y2) / 2 / img_height
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height
                
                # Ensure values are within [0, 1]
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            else:
                print(f"Warning: Unknown class '{label}' in {json_path}")
    
    return yolo_annotations

def setup_yolo_dataset(images_dir, json_dir, output_dir, train_split=0.8):
    """
    Organize images and convert JSON labels to YOLO format
    """
    
    # Create output directory structure
    dirs_to_create = [
        f"{output_dir}/images/train",
        f"{output_dir}/images/val",
        f"{output_dir}/labels/train",
        f"{output_dir}/labels/val"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f"*{ext}"))
        image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle and split
    random.shuffle(image_files)
    train_count = int(len(image_files) * train_split)
    
    train_images = image_files[:train_count]
    val_images = image_files[train_count:]
    
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Define your class mapping based on your labels
    class_mapping = {
        'water_bottle': 0,
        'mouse': 1,
        'airpod_case': 2,
        'mechanical_pencil': 3,
        'keys': 4,
        'guitar_pick': 5,
        'keyboard': 6,
        'monitor': 7, 
        'laptop': 8,
    }
    
    # Process training images
    for img_path in train_images:
        # Copy image
        dst_img = f"{output_dir}/images/train/{img_path.name}"
        shutil.copy2(img_path, dst_img)
        
        # Convert label if exists
        json_file = Path(json_dir) / f"{img_path.stem}.json"
        if json_file.exists():
            # Get image dimensions
            img = cv2.imread(str(img_path))
            img_height, img_width = img.shape[:2]
            
            # Convert JSON to YOLO
            yolo_labels = convert_json_to_yolo(json_file, img_width, img_height, class_mapping)
            
            # Save YOLO label file
            label_file = f"{output_dir}/labels/train/{img_path.stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_labels))
    
    # Process validation images
    for img_path in val_images:
        # Copy image
        dst_img = f"{output_dir}/images/val/{img_path.name}"
        shutil.copy2(img_path, dst_img)
        
        # Convert label if exists
        json_file = Path(json_dir) / f"{img_path.stem}.json"
        if json_file.exists():
            # Get image dimensions
            img = cv2.imread(str(img_path))
            img_height, img_width = img.shape[:2]
            
            # Convert JSON to YOLO
            yolo_labels = convert_json_to_yolo(json_file, img_width, img_height, class_mapping)
            
            # Save YOLO label file
            label_file = f"{output_dir}/labels/val/{img_path.stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_labels))
    
    return class_mapping

def create_dataset_yaml(output_dir, class_mapping):
    """Create dataset.yaml file"""
    
    # Reverse mapping for names
    names = {v: k for k, v in class_mapping.items()}
    
    dataset_config = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': names
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f)
    
    print(f"Created dataset.yaml with {len(names)} classes")
    return dataset_config

def train_model(video_path, output_dir):
    """Train YOLO model with optimized settings"""
    model = YOLO('yolov8n.pt')  # Using nano model for faster training
    
    print("Starting training with optimized settings...")
    print("Using YOLOv8n (nano) for faster training while maintaining good accuracy")

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
    temp_video_path = f"{output_dir}/video_results.mp4"
    rotate_video(video_path, temp_video_path)
    
    results = model.train(
        data='dataset.yaml',
        epochs=100,          # Increased epochs for better learning
        imgsz=640,
        batch=16,            # Increased batch size if your system can handle it
        device='cpu',        # Change to 'cuda' if you have GPU
        project='runs/train',
        name='desk_objects',
        patience=20,         # Early stopping if no improvement
        save_period=10,      # Save checkpoint every 10 epochs
        verbose=True
    )
    
    print(f"Training complete! Best model saved to: {results.save_dir}/weights/best.pt")
    return results


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

def detect_objects_in_video(video_path, model_path=None, output_dir='results'):
    """Run object detection on video and save results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    if model_path and os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"Using custom model: {model_path}")
    else:
        model = YOLO('yolov8l.pt')
        print("Using pre-trained YOLOv8 model")

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
    temp_video_path = f"{output_dir}/video_results.mp4"
    rotate_video(video_path, temp_video_path)
    
    
    # Run detection
    results = model.track(
        source=temp_video_path,
        imgsz=640,
        save=True,                  # writes detection.mp4 for you
        project=output_dir,
        name='detection',
        conf=0.1,
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
    
    # Save to CSV
    if detections:
        df = pd.DataFrame(detections)
        csv_path = f'{output_dir}/detection_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(detections)} detections to {csv_path}")
        
        # Print summary
        print("\nDetection Summary:")
        print(df['object_class'].value_counts())
    
    return csv_path if detections else None

def inspect_json_format(json_file_path):
    """Helper function to inspect your JSON format"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    print("JSON structure:")
    print(json.dumps(data, indent=2)[:500] + "...")
    
    print("\nTop-level keys:", list(data.keys()) if isinstance(data, dict) else "Not a dict")

def main():
    """Main pipeline using your working directory structure""" 
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    # STEP 1: Configure paths using your working structure
    print("\n1. Configuring paths...")
    
    # Your working directory structure
    images_directory = os.path.join(curr_dir, "..", "dataset", "images", "train_upright")
    json_directory = os.path.join(curr_dir, "..", "dataset", "labels", "all")
    output_directory = os.path.join(curr_dir, "..", "dataset")
    video_directory = os.path.join(curr_dir, "..", "assets", "testVideo")
    
    VIDEO_PATH = os.path.join(video_directory, "deskVideo.mp4")
    
    print(f"Images directory: {images_directory}")
    print(f"JSON directory: {json_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Video path: {VIDEO_PATH}")
    
    # Verify paths exist
    if not os.path.exists(images_directory):
        print(f"ERROR: Images directory not found: {images_directory}")
        return
    
    if not os.path.exists(json_directory):
        print(f"ERROR: JSON directory not found: {json_directory}")
        return
    
    # Optional: Inspect JSON format
    print("\n=== Optional: Inspect JSON Format ===")
    sample_json = os.path.join(json_directory, "birdsView_frame_0000.json")
    if os.path.exists(sample_json):
        inspect_json_format(sample_json)
    
    # STEP 2: Setup dataset
    print("\n2. Setting up YOLO dataset...")
    class_mapping = setup_yolo_dataset(
        images_dir=images_directory,
        json_dir=json_directory, 
        output_dir=output_directory,
        train_split=0.8
    )
    
    # Create dataset.yaml
    create_dataset_yaml(output_directory, class_mapping)
    
    print("\n=== Dataset Setup Complete! ===")
    print(f"Classes: {list(class_mapping.keys())}")
    print("Dataset structure:")
    print("├── dataset/")
    print("│   ├── images/")
    print("│   │   ├── train/")
    print("│   │   └── val/")
    print("│   └── labels/")
    print("│       ├── train/")
    print("│       └── val/")
    print("└── dataset.yaml")
    
    # STEP 3: Training options
    print("\n3. Training options:")
    print("a) Train custom model (takes time but better for your specific objects)")
    print("b) Skip training and use pre-trained model (faster)")
    
    choice = input("Choose (a/b): ").lower().strip()
    
    if choice == 'a':
        print("Training custom model...")
        results = train_model(VIDEO_PATH, output_directory)
        model_path = f"{results.save_dir}/weights/best.pt"
        print(f"✓ Custom model trained and saved to: {model_path}")
    else:
        print("Using pre-trained model...")
        model_path = None
    
    if os.path.exists(VIDEO_PATH):
        csv_path = detect_objects_in_video(VIDEO_PATH, model_path)
        if csv_path:
            print(f"Detection CSV file: {csv_path}")
    
    else:   
        # List files in the assets directory to help debug
        video_dir = os.path.dirname(VIDEO_PATH)
        if os.path.exists(video_dir):
            print(f"\nFiles in {video_dir}:")
            for file in os.listdir(video_dir):
                print(f"  - {file}")
        else:
            print(f"\nAssets directory not found: {video_dir}")

if __name__ == "__main__": 
    main()