# YOLO Object Detection Project

This project implements a YOLOv8 object detection model for identifying objects in video and image data.

## Project Structure

The project consists of the following files and directories:

*   `Get_frame.py`: Detects when a delivery box is closed in a video, extracts the frame immediately before the box is closed, and saves the extracted frames to a directory.
*   `YOLO_train.py`: Contains the code for training and evaluating the YOLOv8 model.
*   `YOLO_inference.py`: Contains the code for running inference on video files using the trained YOLOv8 model.
*   `data/`: Contains the data-related files.
    *   `data_preprocessing.py`: Contains the code for preprocessing video and image data.
    *   `raw_data/`: Contains the raw video data.
*   `stable_model/`: Contains the trained model weights and training results.
    *   `best.pt`: The best performing model weights.
    *   `train/`: Contains the training logs, metrics, and visualizations.
*   `requirements.txt`: Lists the project dependencies.
*   `yolov8n.pt`: The pre-trained YOLOv8n model weights.

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
```

## Usage

### 1. Data Preparation

1.  Place your raw video data in the `data/raw_data/` directory.
2.  Run the `data/data_preprocessing.py` script to preprocess the video data into image frames.
    ```bash
    python data/data_preprocessing.py
    ```
    This script will:
    * Extract frames from the videos in `data/raw_data/`.
    * Enhance the images for better object detection (optional).
    * Save the processed images in `data/image_data/<video_name>/`.

### 2. Training

1.  Modify the `YOLO_train.py` script to configure the training parameters, such as the number of epochs, batch size, and learning rate.
2.  Run the `YOLO_train.py` script to train the YOLOv8 model.
    ```bash
    python YOLO_train.py
    ```
    This script will:
    * Split the dataset into training and testing sets.
    * Create a `data.yaml` file that defines the dataset configuration for YOLO.
    * Train the YOLOv8n model using the specified parameters.
    * Evaluate the trained model on the test set.
    * Save the trained model weights and training results in the `stable_model/train/` directory.

### 3. Inference

1.  Modify the `YOLO_inference.py` script to specify the path to the video file you want to run inference on.
2.  Run the `YOLO_inference.py` script to perform object detection on the video.
    ```bash
    python YOLO_inference.py
    ```
    This script will:
    * Load the trained YOLOv8 model weights from `stable_model/best.pt`.
    * Run inference on the specified video file.
    * Display the results with bounding boxes around the detected objects.


## Future Improvements

*   Implement a more robust data preprocessing pipeline.
*   Experiment with different YOLOv8 model architectures and training parameters.
*   Add support for real-time object detection using a webcam.
