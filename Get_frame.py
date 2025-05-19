import cv2
from collections import defaultdict, deque
from ultralytics import YOLO
import pickle
import cv2
import os
from ultralytics import YOLO

model = YOLO("best.pt")
video_path = "data/raw_data/250503_pm_07.13.53_brown_clear_num=4.mp4"
cap = cv2.VideoCapture(video_path)

# output directory for saving cropped images
save_dir = "crops"
os.makedirs(save_dir, exist_ok=True)

# initialize variables
last_class_per_id = {}
frame_id = 0

# set epsilon( epsilon is the number of frames to save before the transition)
epsilon = 5
frame_buffer = deque(maxlen=epsilon)
detection_buffer = deque(maxlen=epsilon)
while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")[0]
    
    boxes, ids, classes = [], [], []
    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)

    # save the current frame and detections in to the buffer
    frame_buffer.append(frame.copy())
    detection_buffer.append({int(i): (b, int(c)) for b, i, c in zip(boxes, ids, classes)})

    for obj_id, cls in zip(ids, classes):
        last_cls = last_class_per_id.get(obj_id)

        # if class 1 → 0，resume ε frames
        if last_cls == 1 and cls == 0:
            print(f"Detected ID {obj_id} class 1→0 at frame {frame_id}, retrieving previous {epsilon} frames")
            for offset, (f_img, dets) in enumerate(zip(frame_buffer, detection_buffer)):
                if obj_id in dets:
                    box, prev_cls = dets[obj_id]
                    x1, y1, x2, y2 = map(int, box)
                    crop = f_img[y1:y2, x1:x2]
                    cv2.imwrite(f"{save_dir}/frame{frame_id - epsilon + offset}_id{obj_id}_cls{prev_cls}.jpg", crop)

        last_class_per_id[obj_id] = cls

    frame_id += 1

cap.release()
print("Done!")
