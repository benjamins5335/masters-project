import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

from models.binary_classifier import BinaryClassifier

# import evaluate.py
from evaluate import evaluate
import os
import argparse
import plotly.graph_objects as go

def train(model, model_file_name, train_loader, val_loader, num_epochs=10, lr=0.0001, weight_decay=0.0, device='cuda', continue_training=False):
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
    
    if 'resnet18' in model_file_name:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
    
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
        if os.path.isfile('{}.pth'.format(model_file_name)):
            print('pth file found. Loading model.')
            model.load_state_dict(torch.load('{}.pth'.format(model_file_name)))
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
        
        val_loss, val_acc, _ = evaluate(model, val_loader, device)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        
        print(f'Train loss: {train_loss / len(train_loader.dataset)}')
        print(f'Train accuracy: {train_acc / len(train_loader.dataset)}')
        print(f'Validation loss: {val_loss}')
        print(f'Validation accuracy: {val_acc}')
        print()


    # Save model
    torch.save(model.state_dict(), 'models/{}.pth'.format(model_file_name))
    

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
    fig.write_image("plots/loss_vs_epochs.png")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=train_acc_history, mode='lines+markers', name='Training Accuracy'))
    fig.add_trace(go.Scatter(x=list(range(1, num_epochs + 1)), y=val_acc_history, mode='lines+markers', name='Validation Accuracy'))
    fig.update_layout(title='Accuracy vs. Epochs', xaxis_title='Epoch', yaxis_title='Accuracy')
    # save the plot
    fig.write_image("plots/accuracy_vs_epochs.png")
                 
                 
                 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    # Add arguments
    parser.add_argument('--continue_training', action='store_true', help='Continue training from a saved model')
    parser.add_argument('--use_pretrained', type=str, help='Use a pretrained model')


    # Parse the arguments
    args = parser.parse_args()
    continue_training = args.continue_training
    pretrained_model = args.use_pretrained
    
    
    with open('config.json') as f:
        config = json.load(f)
    
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    dropout = config['dropout']
    
    key = (learning_rate, batch_size, num_epochs, weight_decay, dropout)
    model_file_name = str(key).replace(' ', '').replace('(', '').replace(')', '').replace(',', '_')
    
    if pretrained_model:
        model_file_name = pretrained_model + '_' + model_file_name
                        

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = ImageFolder(root='data/train', transform=data_transforms)
    
    
    # split the training data into training and validation
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = BinaryClassifier()

    # Train the model
    train_loss_history, train_acc_history, val_loss_history, val_acc_history = train(
        model,
        model_file_name,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=learning_rate,
        device=device,
        continue_training=continue_training
    )
    
    plot_results(num_epochs, train_loss_history, train_acc_history, val_loss_history, val_acc_history)
    
    with open('results.json') as f:
        data = json.load(f)
        # write new data to the file
    data[model_file_name] = {
        'lr': learning_rate,
        'batch_size': batch_size,
        'epochs': num_epochs,
        'weight_decay': weight_decay,
        'dropout': dropout,
        'train_loss_history': train_loss_history,
        'train_acc_history': train_acc_history,
        'val_loss_history': val_loss_history,
        'val_acc_history': val_acc_history
    }
    
    with open('results.json', 'w') as f:
        json.dump(data, f, indent=2)

