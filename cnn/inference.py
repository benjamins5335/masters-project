import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torch.utils.data import DataLoader

from models.binary_classifier import BinaryClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.binary_classifier import BinaryClassifier
from PIL import Image

import streamlit as st

import numpy as np



def inference_cnn(image):
    """
    Loads a trained convolutional neural network and performs inference on a
    single image.

    Returns:
    None
    """
    
    # Load model
    model = BinaryClassifier()
    model.load_state_dict(torch.load('models/0.001_64_30_0.001_0.4.pth'))
    model.eval()
    
    # Convert PIL image to numpy array (assuming image is in RGB mode)
    image_array = np.array(image)
    
    # Downsample the image
    downsampled_image = downsample(image_array, 96, 96)
    
    # transformations
    data_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the input
    ])
    
    # Perform inference
    image_tensor = data_transforms(downsampled_image).unsqueeze(0)
    output = model(image_tensor)
    predicted = torch.sigmoid(output).item()
    
    print(predicted)
    return predicted

def downsample(img, width, height):
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    right = left + crop_size
    top = (height - crop_size) // 2
    bottom = top + crop_size
    
    img = img[top:bottom, left:right]
    
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
    
    return img

if __name__ == "__main__":
    st.title("Binary Classifier Inference App")

    # File uploader widget that accepts only jpg, jpeg, and png extensions
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    

    if uploaded_file is not None:
        # Display the uploaded image
        image_raw = Image.open(uploaded_file)
        
        # Perform inference and display result#
        predicted = inference_cnn(image_raw)
        print(predicted)
        if predicted < 0.5:
            st.write("This is a fake image!")
        else:
            st.write("This is a real image!")
