````markdown
# YOLO Object Detection Project

This project implements a YOLOv8 object detection model for identifying objects in video and image data.

## Project Structure

The project consists of the following files and directories:

*   `.gitignore`
*   `Get_frame.py`: Detects when a delivery box is closed in a video, extracts the frame immediately before the box is closed, and saves the extracted frames to a directory.
*   `README.md`
*   `requirements.txt`: Lists the project dependencies.
*   `YOLO_inference.py`: Contains the code for running inference on video files using the trained YOLOv8 model.
*   `YOLO_lossVisualization.py`
*   `YOLO_train.py`: Contains the code for training and evaluating the YOLOv8 model.
*   `data/`: Contains the data-related files.
    *   `data_preprocessing.py`: Contains the code for preprocessing video and image data.
*   `model/`: Contains the model files.
    *   `4class.pt`
    *   `11class.pt`
    *   `yolov8nOG.pt`

## Dependencies

The project requires the following dependencies:

*   ultralytics
*   cv2
*   os
*   random
*   shutil
*   yaml
*   pandas
*   matplotlib
*   PIL

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
````

## Usage

### 1. Data Preparation

1. Place your raw video data in the `data/raw_data/` directory.

2. Run the `data/data_preprocessing.py` script to preprocess the video data into image frames.

   ```bash
   python data/data_preprocessing.py
   ```

   This script will:

   - Extract frames from the videos in `data/raw_data/`.
   - Enhance the images for better object detection (optional).
   - Save the processed images in `data/image_data/<video_name>/`.

### 2. Training

1. Modify the `YOLO_train.py` script to configure the training parameters, such as the number of epochs, batch size, and learning rate.

2. Run the `YOLO_train.py` script to train the YOLOv8 model.

   ```bash
   python YOLO_train.py
   ```

   This script will:

   - Split the dataset into training and testing sets.
   - Create a `data.yaml` file that defines the dataset configuration for YOLO.
   - Train the YOLOv8n model using the specified parameters.
   - Evaluate the trained model on the test set.
   - Save the trained model weights and training results in the `model/` directory.

### 3. Inference

1. Modify the `YOLO_inference.py` script to specify the path to the video file you want to run inference on.

2. Run the `YOLO_inference.py` script to perform object detection on the video.

   ```bash
   python YOLO_inference.py
   ```

   This script will:

   - Load the trained YOLOv8 model weights from `model/`.
   - Run inference on the specified video file.
   - Display the results with bounding boxes around the detected objects.

## Future Improvements

- Implement a more robust data preprocessing pipeline.
- Experiment with different YOLOv8 model architectures and training parameters.
- Add support for real-time object detection using a webcam.

## Korean Translation

# YOLO 객체 감지 프로젝트

이 프로젝트는 비디오 및 이미지 데이터에서 객체를 식별하기 위한 YOLOv8 객체 감지 모델을 구현합니다.

## 프로젝트 구조

프로젝트는 다음과 같은 파일 및 디렉토리로 구성됩니다.

- `.gitignore`

- `Get_frame.py`: 비디오에서 배달 상자가 닫힐 때를 감지하고, 상자가 닫히기 직전의 프레임을 추출하여 디렉토리에 저장합니다.

- `README.md`

- `requirements.txt`: 프로젝트 종속성을 나열합니다.

- `YOLO_inference.py`: 훈련된 YOLOv8 모델을 사용하여 비디오 파일에서 추론을 실행하기 위한 코드를 포함합니다.

- `YOLO_lossVisualization.py`

- `YOLO_train.py`: YOLOv8 모델을 훈련하고 평가하기 위한 코드를 포함합니다.

- `data/`: 데이터 관련 파일이 들어 있습니다.
  - `data_preprocessing.py`: 비디오 및 이미지 데이터 전처리를 위한 코드를 포함합니다.

- `model/`: 모델 파일이 들어 있습니다.

  - `4class.pt`
  - `11class.pt`
  - `yolov8nOG.pt`

## 종속성

이 프로젝트는 다음 종속성이 필요합니다.

- ultralytics
- cv2
- os
- random
- shutil
- yaml
- pandas
- matplotlib
- PIL

다음 명령을 사용하여 이러한 종속성을 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 데이터 준비

1. 원시 비디오 데이터를 `data/raw_data/` 디렉토리에 넣습니다.

2. `data/data_preprocessing.py` 스크립트를 실행하여 비디오 데이터를 이미지 프레임으로 전처리합니다.

   ```bash
   python data/data_preprocessing.py
   ```

   이 스크립트는 다음을 수행합니다.

   - `data/raw_data/`의 비디오에서 프레임을 추출합니다.
   - 더 나은 객체 감지를 위해 이미지를 향상시킵니다(선택 사항).
   - 처리된 이미지를 `data/image_data/<video_name>/`에 저장합니다.

### 2. 훈련

1. 에포크 수, 배치 크기 및 학습률과 같은 훈련 매개변수를 구성하려면 `YOLO_train.py` 스크립트를 수정하십시오.

2. YOLOv8 모델을 훈련하려면 `YOLO_train.py` 스크립트를 실행하십시오.

   ```bash
   python YOLO_train.py
   ```

   이 스크립트는 다음을 수행합니다.

   - 데이터 세트를 훈련 및 테스트 세트로 분할합니다.
   - YOLO에 대한 데이터 세트 구성을 정의하는 `data.yaml` 파일을 만듭니다.
   - 지정된 매개변수를 사용하여 YOLOv8n 모델을 훈련합니다.
   - 테스트 세트에서 훈련된 모델을 평가합니다.
   - 훈련된 모델 가중치 및 훈련 결과를 `model/` 디렉토리에 저장합니다.

### 3. 추론

1. 추론을 실행할 비디오 파일의 경로를 지정하려면 `YOLO_inference.py` 스크립트를 수정하십시오.

2. 비디오에서 객체 감지를 수행하려면 `YOLO_inference.py` 스크립트를 실행하십시오.

   ```bash
   python YOLO_inference.py
   ```

   이 스크립트는 다음을 수행합니다.

   - `model/`에서 훈련된 YOLOv8 모델 가중치를 로드합니다.
   - 지정된 비디오 파일에서 추론을 실행합니다.
   - 감지된 객체 주위에 경계 상자가 있는 결과를 표시합니다.

## 향후 개선 사항

- 더 강력한 데이터 전처리 파이프라인을 구현합니다.
- 다양한 YOLOv8 모델 아키텍처 및 훈련 매개변수를 실험합니다.
- 웹캠을 사용하여 실시간 객체 감지에 대한 지원을 추가합니다.

## Chinese Translation

# YOLO目标检测项目

本项目实现了一个YOLOv8目标检测模型，用于识别视频和图像数据中的对象。

## 项目结构

本项目由以下文件和目录组成：

- `.gitignore`

- `Get_frame.py`: 检测视频中何时关闭交付箱，提取关闭箱子之前的帧，并将提取的帧保存到目录中。

- `README.md`

- `requirements.txt`: 列出项目依赖项。

- `YOLO_inference.py`: 包含用于使用训练的YOLOv8模型对视频文件运行推理的代码。

- `YOLO_lossVisualization.py`

- `YOLO_train.py`: 包含用于训练和评估YOLOv8模型的代码。

- `data/`: 包含与数据相关的文件。
  - `data_preprocessing.py`: 包含用于预处理视频和图像数据的代码。

- `model/`: 包含模型文件。

  - `4class.pt`
  - `11class.pt`
  - `yolov8nOG.pt`

## 依赖项

本项目需要以下依赖项：

- ultralytics
- cv2
- os
- random
- shutil
- yaml
- pandas
- matplotlib
- PIL

您可以使用pip安装这些依赖项：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据准备

1. 将您的原始视频数据放入`data/raw_data/`目录中。

2. 运行`data/data_preprocessing.py`脚本以将视频数据预处理为图像帧。

   ```bash
   python data/data_preprocessing.py
   ```

   此脚本将：

   - 从`data/raw_data/`中的视频中提取帧。
   - 增强图像以获得更好的对象检测（可选）。
   - 将处理后的图像保存在`data/image_data/<video_name>/`中。

### 2. 训练

1. 修改`YOLO_train.py`脚本以配置训练参数，例如epoch数、批量大小和学习率。

2. 运行`YOLO_train.py`脚本以训练YOLOv8模型。

   ```bash
   python YOLO_train.py
   ```

   此脚本将：

   - 将数据集拆分为训练集和测试集。
   - 创建一个`data.yaml`文件，用于定义YOLO的数据集配置。
   - 使用指定的参数训练YOLOv8n模型。
   - 评估测试集上训练的模型。
   - 将训练的模型权重和训练结果保存在`model/`目录中。

### 3. 推理

1. 修改`YOLO_inference.py`脚本以指定要运行推理的视频文件的路径。

2. 运行`YOLO_inference.py`脚本以对视频执行对象检测。

   ```bash
   python YOLO_inference.py
   ```

   此脚本将：

   - 从`model/`加载训练的YOLOv8模型权重。
   - 对指定的视频文件运行推理。
   - 显示检测到的对象周围带有边界框的结果。

## 未来改进

- 实施更强大的数据预处理管道。
- 试验不同的YOLOv8模型架构和训练参数。
- 添加使用网络摄像头进行实时对象检测的支持。
