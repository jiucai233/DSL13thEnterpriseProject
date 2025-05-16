import cv2
import os
import numpy as np
from datetime import datetime

def preprocess_video(video_path, sample_rate=1, resize_dim=None):
    """
    Preprocesses a video into image frames.

    Args:
    video_path: Path to the video file.
    sample_rate: Sampling rate, save one image every n frames, default is to save every frame.
    resize_dim: Resize image dimensions, format is (width, height), default is no resizing.
    """
    # Extract video name from path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Define output folder
    output_folder = os.path.join("data", "image_data", video_name)

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {frame_count}")
    print(f"Video duration: {duration:.2f} seconds")
    
    # Read and process video frames
    frame_idx = 0
    saved_count = 0
    
    start_time = datetime.now()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Determine whether to save the current frame based on the sampling rate
        if frame_idx % sample_rate == 0:
            # Resize the image (if needed)
            if resize_dim is not None:
                frame = cv2.resize(frame, resize_dim)
            
            # Save the image
            output_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
            # Display progress every 100 frames
            if saved_count % 100 == 0:
                elapsed = (datetime.now() - start_time).seconds
                print(f"Processed {frame_idx} frames, saved {saved_count} images, took {elapsed} seconds")
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    
    print(f"Finished! Processed {frame_idx} frames, saved {saved_count} images")
    return saved_count

def enhance_image_for_detection(image):
    """
    Enhance the image for better lid detection

    Args:
    image: Input image

    Returns:
    Enhanced image
    """
    # Convert to HSV color space, may help distinguish the lid
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:,:,2] = clahe.apply(hsv[:,:,2])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Optional: Denoise
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    return enhanced

def create_detection_dataset(input_folder, output_folder, enhance=True):
    """
    Create a dataset for object detection

    Args:
    input_folder: Input image folder
    output_folder: Output enhanced image folder
    enhance: Whether to enhance the image
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for i, img_file in enumerate(image_files):
        input_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, img_file)
        
        # Read the image
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"Could not read image: {input_path}")
            continue
        
        # Image enhancement
        if enhance:
            image = enhance_image_for_detection(image)
        
        # Save the processed image
        cv2.imwrite(output_path, image)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    print(f"Dataset creation complete, processed {len(image_files)} images")

# Example Usage
if __name__ == "__main__":
    # Video preprocessing example
    i = 1
    video_path = "data/raw_data"  # Replace with your video path
    video_files = [f for f in os.listdir(video_path)]
    for video in os.listdir(video_path):
        print(f"process [{i}/{len(video_files)}]: {video}")
        i+=1
        # 处理视频
        video = os.path.join(video_path, video)
        preprocess_video(video, sample_rate=15, resize_dim=(640, 480))
    
print("all done")
