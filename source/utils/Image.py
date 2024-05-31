"""Module for various tasks regarding images and rendering."""

import cv2
import torch
import numpy as np


def image_to_tensor(image):
    """Converts image to a PyTorch tensor"""
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    
    # if torch.cuda.is_available():
    #     image_tensor = image_tensor.cuda()
    
    return image_tensor

def resize_and_bgr2gray(image):
    # Crop out the floor
    image = image[0:288, 0:404]
    
    # Convert to grayscale and resize image
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    
    return image_data
