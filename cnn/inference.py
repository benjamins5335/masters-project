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


def inference_cnn():
    """
    Loads a trained convolutional neural network and performs inference on a
    single image.

    Returns:
    None
    """
    
    # Load model
    model = BinaryClassifier()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    
    # Load image
    image = Image.open('/home/bs214/Downloads/university_logo.png').convert('RGB')
    
    # transformations
    data_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the input
    ])
    
    # Perform inference
    image_tensor = data_transforms(image).unsqueeze(0)
    output = model(image_tensor)
    predicted = torch.round(output)
    print(f'Predicted: {predicted}')