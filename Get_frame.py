import cv2
from ultralytics import YOLO
import os
import math

def load_video_frames(video_path):
    """载入视频所有帧并返回帧列表及fps"""
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


def detect_transitions(frames, model, tracker_cfg, pre_frames, post_frames):
    """检测每个目标从open->close的帧索引"""
    last_cls = {}
    events = []  # 列表: (obj_id, frame_idx)
    for idx, frame in enumerate(frames):
        results = model.track(frame, persist=True, verbose=False, show=True,tracker="bytetrack.yaml")
        for result in results:
            if result.boxes and result.boxes.id is not None:
                ids = result.boxes.id.numpy().astype(int)
                classes = result.boxes.cls.numpy().astype(int)
                for obj_id, cls in zip(ids, classes):
                    prev = last_cls.get(obj_id)
                    if prev in [7,8,9,10] and cls in [1,2,3,4]:
                        events.append((obj_id, idx))
                    last_cls[obj_id] = cls
    return events


def save_clips(frames, fps, save_dir, events, pre_sec, post_sec):
    """根据事件列表保存每个切片视频"""
    os.makedirs(save_dir, exist_ok=True)
    pre_frames = max(1, math.floor(fps * pre_sec))
    post_frames = max(1, math.floor(fps * post_sec))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames[0].shape[:2]
    for i, (obj_id, idx) in enumerate(events):
        start = max(0, idx - pre_frames)
        end = min(len(frames) - 1, idx + post_frames)
        out_path = os.path.join(save_dir, f"obj{obj_id}_idx{idx}_clip.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for f in frames[start:end+1]:
            writer.write(f)
        writer.release()
        print(f"Saved clip for object {obj_id} at event {idx}: {out_path}")


def file_save(video_path):
    filename = os.path.basename(video_path)
    name_parts = os.path.splitext(filename)[0].split('_')
    output_name = '_'.join(name_parts[3:])
    save_dir = os.path.join("crops", output_name)
    return save_dir


if __name__ == "__main__":
    MODEL_PATH = "11classVerModel/best.pt"
    VIDEO_PATH = "data/raw_data/250504_pm_07.57.52_brown_clear_num=2.mp4"
    PRE_SEC = 2  # 转换前秒数
    POST_SEC = 2  # 转换后秒数
    TRACKER_CFG = "bytetrack.yaml"

    # 加载模型和视频
    model = YOLO(MODEL_PATH)
    frames, fps = load_video_frames(VIDEO_PATH)
    save_dir = file_save(VIDEO_PATH)

    # 检测转换事件
    events = detect_transitions(frames, model, TRACKER_CFG, 
                                pre_frames=max(1, math.floor(fps * PRE_SEC)),
                                post_frames=max(1, math.floor(fps * POST_SEC)))

    # 保存每个对象的剪辑
    save_clips(frames, fps, save_dir, events, PRE_SEC, POST_SEC)
    print("All clips generated.")