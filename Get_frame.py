import cv2
from collections import defaultdict, deque
from ultralytics import YOLO

# 初始化
model = YOLO('best.pt')  # 加载你的多目标检测模型
cap = cv2.VideoCapture("你的视频源")  # 支持文件/RTSP/HTTP

# 为每个盒子维护独立的状态和帧缓冲区
box_buffers = defaultdict(lambda: deque(maxlen=5))  # 格式: {box_id: deque(frames)}
previous_statuses = {}  # 格式: {box_id: "open"或"closed"}

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    results = model(frame, verbose=False)[0]
    
    # 遍历当前帧所有检测到的盒子
    for box in results.boxes:
        box_id = int(box.id[0]) if box.id else None  # 如果模型支持ID追踪，否则用坐标哈希
        if box_id is None:
            box_id = hash(tuple(map(int, box.xyxy[0])))  # 用边界框坐标作为临时ID
        
        current_status = model.names[int(box.cls)]  # 当前盒子状态
        
        # 状态变化检测 (open -> closed)
        if previous_statuses.get(box_id) == "delivery box" and current_status == "closed delivery box":
            if len(box_buffers[box_id]) >= 2:  # 确保有足够的历史帧
                # 保存该盒子闭合前的最后一帧
                target_frame = box_buffers[box_id][-2]
                cv2.imwrite(
                    f"capture_box_{box_id}_at_{frame_count}.jpg", 
                    target_frame[int(box.xyxy[0][1]):int(box.xyxy[0][3]), 
                                int(box.xyxy[0][0]):int(box.xyxy[0][2])]  # 裁剪盒子区域
                )
        
        # 更新该盒子的状态和缓冲区
        previous_statuses[box_id] = current_status
        box_buffers[box_id].append(frame.copy())
    
    frame_count += 1

cap.release()