import pyiqa
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pyiqa import imread2tensor
import numpy as np
import shutil
import subprocess
import os
import sys
import seaborn as sns
from functools import partial
from scipy.stats import spearmanr
import torch.nn as nn
import cv2
import numpy as np
import albumentations


# Function to read YUV 4:2:0 frame and convert it to RGB
def read_yuv_frame(f, width, height):
    frame_size = width * height + (width // 2) * (height // 2) * 2
    yuv_frame = f.read(frame_size)
    if len(yuv_frame) < frame_size:
        return None#, None
    
    # Separate Y, U, and V channels
    Y = np.frombuffer(yuv_frame[0:width*height], dtype=np.uint8).reshape((height, width))
    U = np.frombuffer(yuv_frame[width*height:width*height + (width//2)*(height//2)], dtype=np.uint8).reshape((height//2, width//2))
    V = np.frombuffer(yuv_frame[width*height + (width//2)*(height//2):], dtype=np.uint8).reshape((height//2, width//2))
    
    # Upsample U and V to match Y's resolution
    U_up = cv2.resize(U, (width, height))#, interpolation=cv2.INTER_LINEAR)
    V_up = cv2.resize(V, (width, height))#, interpolation=cv2.INTER_LINEAR)
    
    # Merge Y, U, V into a YUV image and convert to RGB
    yuv_img = cv2.merge([Y, U_up, V_up])
    rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)
    
    return rgb_img, (Y, U, V)

# Function to write YUV 4:2:0 frame
def write_yuv_frame(f, Y, U, V):
    U_down = cv2.resize(U, (Y.shape[1] // 2, Y.shape[0] // 2))#, interpolation=cv2.INTER_LINEAR)
    V_down = cv2.resize(V, (Y.shape[1] // 2, Y.shape[0] // 2))#, interpolation=cv2.INTER_LINEAR)
    f.write(Y.tobytes())
    f.write(U_down.tobytes())
    f.write(V_down.tobytes())


def parse_psnr_avg(file_path):
    psnr_values = []

    with open(file_path, 'r') as file:
        for line in file:
            try:
                val = float(line.split('psnr_avg:')[1].split()[0])
            except Exception:
                val = 1000
            psnr_values.append(val)

    if psnr_values:
        mean_psnr = sum(psnr_values) / len(psnr_values)
        return mean_psnr
    else:
        return None
