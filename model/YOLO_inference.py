import cv2
from ultralytics import YOLO
import os
import math
from collections import deque
import pickle
import shutil


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


def detect_transitions(frames, model, close_index, open_index, tracker_cfg="bytetrack.yaml", conf=0.8,):
    """Detect transitions and optionally save detection results."""
    last_cls = {}
    events = []

    for idx, frame in enumerate(frames):
        results = model.track(frame, persist=True, verbose=False, conf=conf, tracker=tracker_cfg)

        for result in results:
            if result.boxes and result.boxes.id is not None:
                ids = result.boxes.id.numpy().astype(int)
                classes = result.boxes.cls.numpy().astype(int)
                for obj_id, cls in zip(ids, classes):
                    prev = last_cls.get(obj_id)
                    if prev in open_index and cls in close_index:
                        events.append((obj_id, idx))
                    last_cls[obj_id] = cls

    return events,results


def save_clips(frames, fps, save_dir, events, pre_sec, post_sec):
    """save clips based on the frames"""
    out_paths = []
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
        out_path = os.path.join(clip_dir, f'obj{obj_id}_idx{idx}_clip.mp4')
        out_paths.append(out_path)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for f in frames[start:end+1]:
            writer.write(f)
        writer.release()
        print(f"Saved clip for object {obj_id} at event {idx}: {out_path}")
        
    return out_paths


def make_dir(video_path):
    filename = os.path.basename(video_path)
    name_parts = os.path.splitext(filename)[0].split('_')
    output_name = '_'.join(name_parts[3:])
    save_dir = os.path.join("crops", output_name)
    return save_dir

def copy_annotated_frames(save_dir, frame_ids, annotated_subdir="annotated_frames"):
    source_folder = os.path.join(save_dir, annotated_subdir)
    os.makedirs(source_folder, exist_ok=True)  # 혹시 몰라

    for fno in frame_ids:
        # track() 에서 저장된 이름 (네 자리 패딩)
        src_fname = f"frame{fno:04d}.jpg"
        src_path  = os.path.join(source_folder, src_fname)
        if not os.path.isfile(src_path):
            print(f" {src_fname} not found, skipping")
            continue

        dst_fname = f"annotated_frame{fno:04d}.jpg"
        dst_path  = os.path.join(save_dir, dst_fname)
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_fname} → {dst_fname}")

def extract_and_copy_best_frames(
    video_path: str,
    model_path: str,
    save_dir: str,
    open_index: list,
    close_index: list,
    food_classes: set,
    blur_thresh: float = 600.0,
    conf: float = 0.6,
    buffer_size: int = 20
):
    os.makedirs(save_dir, exist_ok=True)
    model = YOLO(model_path)

    # 1) Track 전체 비디오
    track_results = model.track(
        source=video_path,
        conf=conf,
        save=True,
        project=save_dir,
        name="annotated_frames",
        tracker="bytetrack.yaml",
        persist=True,
        stream=False,
        save_frames=True,
        save_txt=True,
        verbose=False
    )
    print(f" → Got {len(track_results)} frame results")

    # 2) 원본 비디오 열기 & 버퍼 준비
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_buffer      = deque(maxlen=buffer_size)
    detect_buffer     = deque(maxlen=buffer_size)
    last_class_per_id = {}
    frame_idx         = 0
    selected_ids      = []

    def focus_measure_laplacian(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    # 3) 이벤트 감지 및 raw 프레임 저장
    for res in track_results:
        ret, frame = cap.read()
        if not ret:
            break

        if res.boxes.id is not None:
            ids     = res.boxes.id.cpu().numpy().astype(int)
            classes = res.boxes.cls.cpu().numpy().astype(int)
            boxes   = res.boxes.xyxy.cpu().numpy().astype(int)
            dets    = {int(i):(tuple(b), int(c)) for b,i,c in zip(boxes, ids, classes)}
        else:
            dets = {}

        frame_buffer.append(frame.copy())
        detect_buffer.append(dets)

        for obj_id, (box, cls) in dets.items():
            prev = last_class_per_id.get(obj_id)
            # 3) 파라미터 이름에 맞춰 조건문 수정
            if prev is not None and prev in open_index and cls in close_index:
                print(f"[Event] ID={obj_id} at frame {frame_idx}: {prev}→{cls}")

                best = {"area":0, "idx":None}
                for bi, dd in enumerate(detect_buffer):
                    for _, (b, ocls) in dd.items():
                        if ocls not in food_classes:
                            continue
                        x1,y1,x2,y2 = b
                        area = (x2-x1)*(y2-y1)
                        if area > best["area"]:
                            best.update({"area":area, "idx":bi})

                if best["idx"] is None:
                    print(" → Warning: no FOOD_CLASSES in buffer")
                else:
                    sel_idx   = best["idx"]
                    sel_frame = frame_buffer[sel_idx]
                    blur_val  = focus_measure_laplacian(sel_frame)
                    while blur_val < blur_thresh and sel_idx > 0:
                        sel_idx   -= 1
                        sel_frame = frame_buffer[sel_idx]
                        blur_val  = focus_measure_laplacian(sel_frame)

                    fno     = frame_idx - len(frame_buffer) + sel_idx + 1
                    fno_str = f"{fno:04d}"
                    filename = f"frame{fno_str}_area{best['area']}_blur{blur_val:.1f}.jpg"
                    path     = os.path.join(save_dir, filename)
                    cv2.imwrite(path, sel_frame)
                    print(f" → Saved raw: {filename}")

                    selected_ids.append(fno)

            last_class_per_id[obj_id] = cls

        frame_idx += 1

    cap.release()

    # 4) annotated 복사
    copy_annotated_frames(save_dir, selected_ids)

    print("All done!")


if __name__ == "__main__":
    # basic config
    MODEL_PATH   = "model/9class.pt"
    VIDEO_PATH   = "data/raw_data/250504_pm_07.57.52_brown_clear_num=2.mp4"
    SOURCE_PATH  = ""
    PRE_SEC      = 2
    POST_SEC     = 2
    OPEN_INDEX   = [5,6,7,8]
    CLOSE_INDEX  = [0,1,2,3]
    FOOD_CLASSES = {4}

    # prep
    frames, fps = load_video_frames(VIDEO_PATH)
    save_dir    = make_dir(VIDEO_PATH)

    # detect transitions
    events,_ = detect_transitions(
        frames, model=YOLO(MODEL_PATH),
        close_index=CLOSE_INDEX,
        open_index =OPEN_INDEX,
        conf=0.8,
        tracker_cfg="bytetrack.yaml",
    )
    # save clips
    clip_paths = save_clips(frames, fps, save_dir, events, PRE_SEC, POST_SEC)

    # extract best frames + copy annotated
    for clip in clip_paths:
        extract_and_copy_best_frames(
        video_path = clip, 
        model_path   = MODEL_PATH,
        save_dir     = save_dir,
        open_index   = OPEN_INDEX,
        close_index  = CLOSE_INDEX,
        food_classes = FOOD_CLASSES,
        blur_thresh  = 600.0,
        conf         = 0.6,
        buffer_size  = 20
    )

    print("All clips generated.")