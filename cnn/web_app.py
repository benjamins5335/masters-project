
import os
import cv2
import torch
import torchvision.transforms as transforms
from models.binary_classifier import BinaryClassifier
import streamlit as st
import numpy as np
from PIL import Image


def inference_individual(image):
    """Performs inference on a single image.

    Args:
        image (Image): The image to perform inference on.

    Returns:
        float: Sigmoid output of the model's prediction.
    """
    
    # Load model
    model = BinaryClassifier()
    model.load_state_dict(torch.load('models/model.pth', map_location=torch.device('cpu')))
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
    with torch.no_grad():
        output = model(image_tensor)
        
    predicted = torch.sigmoid(output)
    
    return predicted


def downsample(img, width, height):
    """Downsamples the image to a square image of size (width, height).

    Args:
        img (np.ndarray): The image to downsample.
        width (int): The width of the downsampled image.
        height (int): The height of the downsampled image.

    Returns:
        np.ndarray: The downsampled image.
    """
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    right = left + crop_size
    top = (height - crop_size) // 2
    bottom = top + crop_size
    
    img = img[top:bottom, left:right]
    
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
    
    return img


def streamlit_app():
    """Creates the web app using Streamlit.
    """
    
    st.title("Binary Classifier Inference App")

    # File uploader widget that accepts only jpg, jpeg, and png extensions
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image_raw = Image.open(uploaded_file).convert('RGB')
        width, height = image_raw.size
        height_ratio = height / 400
        
        # st.image(image_raw, width=int(width / height_ratio))
        
        centre_column = st.columns([1,3,1])[1]
        with centre_column:        
            st.image(image_raw, width=int(width / height_ratio))     
        
        # Perform inference and display result
        predicted = inference_individual(image_raw)
        if predicted < 0.5:
            probablility = 1 - predicted
            st.header("I am {:.2%} sure that this is a fake image!".format(probablility.item()))
        else:
            st.header("I am {:.2%} sure that this is a real image!".format(predicted.item()))



if __name__ == "__main__":
    streamlit_app()