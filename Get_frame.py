import cv2
from ultralytics import YOLO
import os
import math

def load_video_frames(video_path):
    """load all the frames in video and return frame list and fps"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def detect_transitions(frames, model, close_index , open_index , tracker_cfg="bytetrack.yaml" , conf=0.8):
    """detect every object from open->close"""
    last_cls = {}
    events = []  # list: (obj_id, frame_idx)
    for idx, frame in enumerate(frames):
        results = model.track(frame, 
                              persist=True, 
                              verbose=False,
                              conf=conf, 
                              tracker=tracker_cfg)
        for result in results:
            if result.boxes and result.boxes.id is not None:
                ids = result.boxes.id.numpy().astype(int)
                classes = result.boxes.cls.numpy().astype(int)
                for obj_id, cls in zip(ids, classes):
                    prev = last_cls.get(obj_id)
                    if prev in open_index and cls in close_index:
                        events.append((obj_id, idx))
                    last_cls[obj_id] = cls
    return events


def save_clips(frames, fps, save_dir, events, pre_sec, post_sec):
    """save clips based on the frames"""
    os.makedirs(save_dir, exist_ok=True)
    pre_frames = max(1, math.floor(fps * pre_sec))
    post_frames = max(1, math.floor(fps * post_sec))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames[0].shape[:2]
    for i, (obj_id, idx) in enumerate(events):
        start = max(0, idx - pre_frames)
        end = min(len(frames) - 1, idx + post_frames)
        clip_dir = os.path.join(save_dir, f'obj{obj_id}_idx{idx}')
        os.makedirs(clip_dir, exist_ok=True)
        out_path = os.path.join(clip_dir, "clip.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for f in frames[start:end+1]:
            writer.write(f)
        writer.release()
        print(f"Saved clip for object {obj_id} at event {idx}: {out_path}")


def make_dir(video_path):
    filename = os.path.basename(video_path)
    name_parts = os.path.splitext(filename)[0].split('_')
    output_name = '_'.join(name_parts[3:])
    save_dir = os.path.join("crops", output_name)
    return save_dir


if __name__ == "__main__":
    #basic config
    MODEL_PATH = "model/11class.pt"
    VIDEO_PATH = "data/raw_data/250504_pm_07.57.52_brown_clear_num=2.mp4"
    TRACKER_CFG = "bytetrack.yaml"

    #funtion configuration
    PRE_SEC = 2 #second(s) to save before the close event
    POST_SEC = 2  #..after the close event
    OPEN_INDEX = [7,8,9,10] # model's index of opened box
    CLOSE_INDEX = [1,2,3,4] # ...of closed box
    
    # load model and video
    model = YOLO(MODEL_PATH)
    frames, fps = load_video_frames(VIDEO_PATH)
    save_dir = make_dir(VIDEO_PATH)

    # detect close lid event
    events = detect_transitions(frames, model, CLOSE_INDEX, OPEN_INDEX,TRACKER_CFG)

    # save all the videos of event
    save_clips(frames, fps, save_dir, events, PRE_SEC, POST_SEC)
    print("All clips generated.")