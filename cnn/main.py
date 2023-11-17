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


def train(model, train_loader, test_loader, num_epochs=10, lr=0.0001, device='cuda'):
    """
    Trains a convolutional neural network on a binary classification task.

    Args:
    - model: The model to be trained
    - train_loader: DataLoader for training data
    - test_loader: DataLoader for testing data
    - num_epochs: Number of training epochs
    - lr: Learning rate for the optimizer
    - device: Device to use for training ('cuda' or 'cpu')

    Returns:
    None
    """
    
    # implement grid search for hyperparameter tuning
    # define the grid search parameters
    batch_size = [16, 32, 64]
    epochs = [10, 20, 30]
    learning_rate = [0.0001, 0.001, 0.01]
    
    
    # Set device
    model = model.to(device)
    
    # Initialize loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        model.train()
        train_loss = 0
        train_acc = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            train_acc += (predicted == labels).sum().item()

        print(f'Train loss: {train_loss / len(train_loader.dataset)}')
        print(f'Train accuracy: {train_acc / len(train_loader.dataset)}')

        evaluate(model, test_loader, device)

    # Save model
    torch.save(model.state_dict(), 'model.pth')
    
               

def evaluate(model, data_loader, device='cuda'):
    """
    Evaluates a model on a given dataset.

    Args:
    - model: The model to be evaluated
    - data_loader: DataLoader for the evaluation data
    - device: Device to use for evaluation ('cuda' or 'cpu')

    Returns:
    None
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    test_loss = 0
    test_acc = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device).float()
        outputs = model(images).squeeze()
        loss = loss_fn(outputs, labels)
        test_loss += loss.item()
        predicted = torch.round(torch.sigmoid(outputs))
        test_acc += (predicted == labels).sum().item()

    print(f'Test loss: {test_loss / len(data_loader.dataset)}')
    print(f'Test accuracy: {test_acc / len(data_loader.dataset)}')
    print()

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
    BATCH_SIZE = 16

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_ds = ImageFolder(root='data', transform=data_transforms)
    train_size = int(0.8 * len(full_ds))
    test_size = len(full_ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = BinaryClassifier()

    # Train the model
    train(model, train_loader, test_loader, num_epochs=10, lr=0.0001, device=device)