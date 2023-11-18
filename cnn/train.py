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

# import evaluate.py
from evaluate import evaluate
import os
import argparse
import plotly.graph_objects as go



def train(model, train_loader, val_loader, num_epochs=10, lr=0.0001, device='cuda', continue_training=False):
    """
    Trains a convolutional neural network on a binary classification task.

    Args:
    - model: The model to be trained
    - train_loader: DataLoader for training data
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

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    if continue_training:
        if os.path.isfile('model.pth'):
            model.load_state_dict(torch.load('model.pth'))
            print('Loaded model.pth')
        else:
            print('model.pth not found. Training from scratch.')

    

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

        # get average loss and accuracy for the epoch
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(avg_train_acc)
        
        val_loss, val_acc = evaluate(model, val_loader, device)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        
        print(f'Train loss: {train_loss / len(train_loader.dataset)}')
        print(f'Train accuracy: {train_acc / len(train_loader.dataset)}')
        print(f'Validation loss: {val_loss}')
        print(f'Validation accuracy: {val_acc}')
        print()


    # Save model
    torch.save(model.state_dict(), 'model.pth')
    

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


def plot_results(num_epochs, train_loss_history, train_acc_history, val_loss_history, val_acc_history):
    """
    Plots the training and validation loss and accuracy for each epoch.

    Args:
    - num_epochs: Number of epochs
    - train_loss_history: List containing the training loss for each epoch
    - train_acc_history: List containing the training accuracy for each epoch
    - val_loss_history: List containing the validation loss for each epoch
    - val_acc_history: List containing the validation accuracy for each epoch

    Returns:
    None
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=train_loss_history, mode='lines+markers', name='Training Loss'))
    fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=val_loss_history, mode='lines+markers', name='Validation Loss'))
    fig.update_layout(title='Loss vs. Epochs', xaxis_title='Epoch', yaxis_title='Loss')
    fig.show()
    fig.write_image("plots/loss_vs_epochs.png")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=train_acc_history, mode='lines+markers', name='Training Accuracy'))
    fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=val_acc_history, mode='lines+markers', name='Validation Accuracy'))
    fig.update_layout(title='Accuracy vs. Epochs', xaxis_title='Epoch', yaxis_title='Accuracy')
    fig.show()
    # save the plot
    fig.write_image("plots/accuracy_vs_epochs.png")
                 
                 
                 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    os.mkdir('plots')

    
    # Add arguments
    parser.add_argument('--continue_training', action='store_true', help='Continue training from a saved model')

    # Parse the arguments
    args = parser.parse_args()
    continue_training = args.continue_training
    BATCH_SIZE = 16

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = ImageFolder(root='data/train', transform=data_transforms)
    test_ds = ImageFolder(root='data/test', transform=data_transforms)
    
    
    # split the training data into training and validation
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = BinaryClassifier()

    # Train the model
    train_loss_history, train_acc_history, val_loss_history, val_acc_history = train(
        model,
        train_loader,
        val_loader,
        num_epochs=10,
        lr=0.0001,
        device=device,
        continue_training=continue_training
    )
    
    plot_results(10, train_loss_history, train_acc_history, val_loss_history, val_acc_history)
    
    