import os
import random
import shutil
import yaml
from ultralytics import YOLO

# Define the paths
DATA_DIR = 'data/labelled_data4OBJ'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
IMAGE_DIR = 'images'
LABEL_DIR = 'labels'

# Set the split ratio
TRAIN_RATIO = 0.8

# Define the classes
CLASSES = ['closed delivery box','delivery box','food', 'hand']

def split_data():
    # Create the train and test directories if they don't exist
    os.makedirs(os.path.join(TRAIN_DIR, IMAGE_DIR), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_DIR, LABEL_DIR), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, IMAGE_DIR), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, LABEL_DIR), exist_ok=True)

    # Get the list of all image files
    image_files = []
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            if filename.endswith('.jpg'):
                image_files.append(os.path.join(dirpath, filename))

    # Shuffle the image files
    random.shuffle(image_files)

    # Split the image files into train and test sets
    train_size = int(len(image_files) * TRAIN_RATIO)
    train_files = image_files[:train_size]
    test_files = image_files[train_size:]

    # Copy the train files to the train directory
    for image_file in train_files:
        label_file = image_file.replace(IMAGE_DIR, LABEL_DIR).replace('.jpg', '.txt')
        dest_image_file = os.path.join(TRAIN_DIR, IMAGE_DIR, os.path.basename(image_file))
        dest_label_file = os.path.join(TRAIN_DIR, LABEL_DIR, os.path.basename(label_file))
        if not os.path.exists(dest_image_file):
            shutil.copy(image_file, dest_image_file)
        if os.path.exists(label_file) and not os.path.exists(dest_label_file):
            shutil.copy(label_file, dest_label_file)
        else:
            print(f"Warning: No label file found for {image_file}")

    # Copy the test files to the test directory
    for image_file in test_files:
        label_file = image_file.replace(IMAGE_DIR, LABEL_DIR).replace('.jpg', '.txt')
        dest_image_file = os.path.join(TEST_DIR, IMAGE_DIR, os.path.basename(image_file))
        dest_label_file = os.path.join(TEST_DIR, LABEL_DIR, os.path.basename(label_file))
        if not os.path.exists(dest_image_file):
            shutil.copy(image_file, dest_image_file)
        if os.path.exists(label_file) and not os.path.exists(dest_label_file):
            shutil.copy(label_file, dest_label_file)
        else:
            print(f"Warning: No label file found for {image_file}")

def create_data_yaml():
    # Create the data.yaml file
    data = {
        'train': os.path.abspath(os.path.join(DATA_DIR, 'train')),
        'val': os.path.abspath(os.path.join(DATA_DIR, 'test')),
        'nc': len(CLASSES),
        'names': CLASSES
    }

    with open('data.yaml', 'w') as f:
        yaml.dump(data, f)

def train_model():
    # Load the YOLOv8n model
    model = YOLO('model/yolov8nOG.pt')

    # Train the model
    model.train(data='data.yaml', epochs=100)

def eval_model():
    # Load the trained model
    model = YOLO('runs/detect/train/weights/best.pt')

    # Evaluate the model
    metrics = model.val(data='data.yaml')

if __name__ == '__main__':
    split_data()
    create_data_yaml()
    train_model()
    eval_model()
