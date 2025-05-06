import os
import random
import shutil
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO('D:/my_codes/DSL13thEnterpriseProject/runs/detect/train/weights/best.pt')#change!!
results = model.predict(
    source='D:/my_codes/DSL13thEnterpriseProject/data/image_data/250504_pm_08.52.28_brown_clear_num=4/frame_001980.jpg',#change!!
    show=True,  
    conf=0.5,   
)
annotated_img = results[0].plot()


plt.imshow(annotated_img[:, :, ::-1])  # BGR2RGB
plt.axis('off')
plt.show()



# # Plot the training and validation loss
# # Load the training results from the CSV file
# # Note: Make sure to change the path to your results.csv file
# data = pd.read_csv("D:/my_codes/DSL13thEnterpriseProject/runs/detect/train/results.csv")
# plt.figure(figsize=(10, 6))
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.plot(data['epoch'], data['val/box_loss'], label='val/box_loss', marker='x')
# plt.plot(data['epoch'], data['val/cls_loss'], label='val/cls_loss', marker='x')
# plt.plot(data['epoch'], data['train/box_loss'], label='train/box_loss', marker='o')
# plt.plot(data['epoch'], data['train/cls_loss'], label='train/cls_loss', marker='o')
# plt.legend()
# plt.show()