import os
import random
import shutil
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
# this section is for testing the model on a video
model = YOLO('best.pt') #change!!
results = model(
    source='data/raw_data/250503_pm_07.13.53_brown_clear_num=4.mp4', #change to your video path or if you want to use webcam, use '0'
    save=True, #save the results
    # show=True, #instantly show the results
    conf=0.5,   
)
