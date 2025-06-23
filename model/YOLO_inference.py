import cv2
from ultralytics import YOLO
import os
import math
from collections import deque
import shutil

def load_video_frames(video_path):
    """Load all the frames from a video and return the frame list and FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

def detect_transitions(frames, model, close_index, open_index):
    """Detect transitions between open and close states using the YOLO model."""
    last_cls = {}
    events = []
    results = []

    for idx, frame in enumerate(frames):
        current_results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
        results.extend(current_results)

        for result in current_results:
            if result.boxes and result.boxes.id is not None:
                ids = result.boxes.id.numpy().astype(int)
                classes = result.boxes.cls.numpy().astype(int)
                for obj_id, cls in zip(ids, classes):
                    prev = last_cls.get(obj_id)
                    if prev in open_index and cls in close_index:
                        events.append((obj_id, idx))
                    last_cls[obj_id] = cls

    return events, results

def save_event_folders(frames, fps, save_dir, events, pre_sec, post_sec, model):
    """Save video clips and best frames for each event."""
    os.makedirs(save_dir, exist_ok=True)
    pre_frames = max(1, math.floor(fps * pre_sec))
    post_frames = max(1, math.floor(fps * post_sec))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames[0].shape[:2]

    for i, (obj_id, idx) in enumerate(events):
        event_dir = os.path.join(save_dir, f'event_{i+1}_obj{obj_id}_frame{idx}')
        os.makedirs(event_dir, exist_ok=True)
        raw_dir = os.path.join(event_dir, 'raw')
        annotated_dir = os.path.join(event_dir, 'annotated')
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(annotated_dir, exist_ok=True)

        # Save raw video clip
        start = max(0, idx - pre_frames)
        end = min(len(frames) - 1, idx + post_frames)
        raw_video_path = os.path.join(raw_dir, 'clip_raw.mp4')
        writer = cv2.VideoWriter(raw_video_path, fourcc, fps, (w, h))
        for f in frames[start:end+1]:
            writer.write(f)
        writer.release()

        # Save best frame before the event in raw
        best_frame_idx = max(0, idx - 1)
        best_frame = frames[best_frame_idx]
        raw_frame_path = os.path.join(raw_dir, f'best_frame_{best_frame_idx:04d}.jpg')
        cv2.imwrite(raw_frame_path, best_frame)

        # Generate and save annotated frame
        annotated_frame_path = os.path.join(annotated_dir, f'best_frame_{best_frame_idx:04d}_annotated.jpg')
        results = model(best_frame)
        annotated_img = results[0].plot()
        cv2.imwrite(annotated_frame_path, annotated_img)

        # Generate and save annotated video clip
        annotated_video_path = os.path.join(annotated_dir, 'clip_annotated.mp4')
        writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (w, h))
        for f in frames[start:end+1]:
            results = model(f)
            annotated_f = results[0].plot()
            writer.write(annotated_f)
        writer.release()

        print(f"Created event folder {i+1} for object {obj_id} at frame {idx}")

    return save_dir

if __name__ == "__main__":
    MODEL_PATH = "model/4class.pt"
    VIDEO_PATH = "data/raw_data/250523_am_12.23.58_brown_clear_num=2.mp4"
    SAVE_DIR = "output_events"
    PRE_SEC = 2
    POST_SEC = 2
    OPEN_INDEX = [1]  # Assuming class 1 is open box
    CLOSE_INDEX = [0]  # Assuming class 0 is closed box

    # Load video frames
    frames, fps = load_video_frames(VIDEO_PATH)

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Detect transitions
    events, _ = detect_transitions(frames, model, CLOSE_INDEX, OPEN_INDEX)

    if not events:
        print("No events detected. Exiting.")
        exit(1)

    # Save event folders with raw and annotated content
    save_event_folders(frames, fps, SAVE_DIR, events, PRE_SEC, POST_SEC, model)

    print("All done!")