## 한국어: YOLO 객체 탐지 프로젝트

이 프로젝트는 YOLOv8 객체 탐지 모델을 구현하여 비디오 및 이미지 데이터에서 객체를 식별합니다.

### 프로젝트 구조

다음 파일과 디렉토리로 구성되어 있습니다:

- `.gitignore`  
- `README.md`  
- `requirements.txt`: 프로젝트 의존성 목록  
- `YOLO_inference.py`: 학습된 YOLOv8 모델로 비디오 파일에서 추론  
- `YOLO_lossVisualization.py`  
- `YOLO_train.py`: 모델 학습 및 평가 코드  
- `data/`: 데이터 관련 스크립트  
  - `data_preprocessing.py`: 비디오 및 이미지 데이터 전처리  
- `model/`: 모델 파일  
  - `4class.pt`: 닫힌 박스, 열린 박스, 음식, 손을 구분  
  - `11class.pt`: 일반/빨간/검은/흰 박스를 열림/닫힘 상태로 구분 + 음식: 일반/매운 치킨, 떡볶이  
  - `9class.pt`: 11class 모델보다 "음식"은 로 구분  
  - `yolov8nOG.pt`: 원본 YOLOv8n 모델  

### 🧩 의존성

필요한 패키지:

- ultralytics  
- cv2  
- lap  
- pandas  
- matplotlib  

설치:

```bash
pip install -r requirements.txt
```

### 🛠️ 사용법

#### 1. 데이터 준비

1. `data/raw_data/` 디렉토리에 원시 비디오 넣기  
2. 다음 명령어 실행:

```bash
python data/data_preprocessing.py
```

- 프레임 추출  
- 이미지 향상 (선택 사항)  
- `data/image_data/<video_name>/`에 저장  

3. `data/labelled_data4OBJ/` 폴더 생성 및 YOLO 형식 라벨링:

```
labelled_data4OBJ/
  ├── images/
  ├── labels/
  ├── classes.txt
  └── notes.json
```

#### 2. 학습

1. `YOLO_train.py` 파일에서 학습 파라미터 수정  
2. 다음 명령어로 학습 실행:

```bash
python YOLO_train.py
```

- 데이터셋 분할  
- `data.yaml` 생성  
- 모델 학습 및 평가  
- 결과는 `model/`에 저장  

#### 3. 추론

1. `YOLO_inference.py`에서 비디오 경로 설정  
2. 다음 명령어 실행:

```bash
python YOLO_inference.py
```

- 모델 로드  
- 비디오 객체 탐지 및 결과 시각화  