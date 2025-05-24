# YOLO Object Detection Project

This project implements a YOLOv8 object detection model for identifying objects in video and image data.

## Project Structure

The project consists of the following files and directories:

*   `.gitignore`
*   `README.md`
*   `requirements.txt`: Lists the project dependencies.
*   `YOLO_inference.py`: Contains the code for running inference on video files using the trained YOLOv8 model.
*   `YOLO_lossVisualization.py`
*   `YOLO_train.py`: Contains the code for training and evaluating the YOLOv8 model.
*   `data/`: Contains the data-related files.
    *   `data_preprocessing.py`: Contains the code for preprocessing video and image data.
*   `model/`: Contains the model files.
    *   `4class.pt`the model that can distinguish - closed box, open box, foods and hands
    *   `11class.pt`the model that can distinguish - normal box, red box, black box and white box. Meanwhile those boxes can be distinguished by open/closed. Also, the food can being distinguished by normal/spicy Chiken and Tokbbokki
    *   `9class.pt`compared to the 11class one, this one can only distinguish the food to food
    *   `yolov8nOG.pt`original yolov8n model

## Dependencies

The project requires the following dependencies:

*   ultralytics
*   cv2
*   lap
*   pandas
*   matplotlib

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

3. create a`labelled_data4OBJ` folder under the `data` directory. Label the images in YOLO format, and put them in `labelled_data4OBJ/`. Which should contain the following structure: 
```
labelled_data4OBJ/
  ├── images/
  ├── labels/
  ├── classes.txt
  └── notes.json
```

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

---
Created by Yonsei Data Science Lab.
