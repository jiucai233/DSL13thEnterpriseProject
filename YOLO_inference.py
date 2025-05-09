import os
import random
import shutil
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# # this section is for testing the model on a video
# model = YOLO('runs/detect/train/weights/best.pt') #change!!
# results = model.predict(
#     source='data/raw_data/clear_vid_test_box2.mp4', #change!!
#     show=True,
#     save=True,  
#     conf=0.5,   
# )

# Plot the training and validation loss
# Load the training results from the CSV file
# Note: Make sure to change the path to your results.csv file
data = pd.read_csv("runs/detect/train/results.csv")
plt.figure(figsize=(10, 6))
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(data['epoch'], data['val/box_loss'], label='val/box_loss', marker='x')
plt.plot(data['epoch'], data['val/cls_loss'], label='val/cls_loss', marker='x')
plt.plot(data['epoch'], data['train/box_loss'], label='train/box_loss', marker='o')
plt.plot(data['epoch'], data['train/cls_loss'], label='train/cls_loss', marker='o')
plt.legend()
plt.show()