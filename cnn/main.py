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

def train_cnn():
    """
    Trains a convolutional neural network on a binary classification task.

    Returns:
    None
    """
    
    BATCH_SIZE = 16
    
    # Define data transformations (similar to ImageDataGenerator)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # used from imagenet
    ])
    
    full_ds = ImageFolder(root='data', transform=data_transforms)
    
    train_size = int(0.8 * len(full_ds))
    test_size = len(full_ds) - train_size
    
    train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Set device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Initialize model, loss function, and optimizer
    model = BinaryClassifier()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    

    model = model.to(device)

    # Train model
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        model.train()
        train_loss = 0
        train_acc = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()  # Ensure labels are floats
            optimizer.zero_grad()
            outputs = model(images).squeeze()  # Squeeze the output to remove the extra dimension
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))  # Apply sigmoid and round to get predictions
            train_acc += (predicted == labels).sum().item()

            
        print(f'Train loss: {train_loss / len(train_ds)}')
        print(f'Train accuracy: {train_acc / len(train_ds)}')
        
        model.eval()
        test_loss = 0
        test_acc = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images).squeeze()
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            test_acc += (predicted == labels).sum().item()
            
        print(f'Test loss: {test_loss / len(test_ds)}')
        print(f'Test accuracy: {test_acc / len(test_ds)}')
        print()
        

    # Save model
    torch.save(model.state_dict(), 'model.pth')


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
    
if __name__ == '__main__':
    # train_cnn()
    inference_cnn()