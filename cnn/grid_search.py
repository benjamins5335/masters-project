import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models.binary_classifier import BinaryClassifier

from train import train, plot_results
from evaluate import evaluate
import os
import json


def grid_search():
    """
    Test all hyperparameters for the CNN model.
    """

    # define the hyperparameter grid
    param_grid = {
        'lr': [0.0001],
        'batch_size': [32],
        'epochs': [50],
        'weight_decay': [0, 0.001, 0.01],
        'dropout': [.65]
    }
    
    # standard ImageNet normalization values
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    # nested for loops to test all hyperparameters
    for lr in param_grid['lr']:
        for batch_size in param_grid['batch_size']:
            for epochs in param_grid['epochs']:
                for weight_decay in param_grid['weight_decay']:                    
                    for dropout in param_grid['dropout']:
                        key = (lr, batch_size, epochs, weight_decay, dropout)
                        model_file_name = str(key).replace(' ', '').replace('(', '').replace(')', '').replace(',', '_')
                        
                        if os.path.exists(f'models/{model_file_name}.pth'):
                            continue
                        
                        print(f'lr: {lr}, batch_size: {batch_size}, epochs: {epochs}, weight_decay: {weight_decay}, dropout: {dropout}')
                        model = BinaryClassifier(dropout=dropout)
                        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

                        train_loss_history, train_acc_history, val_loss_history, val_acc_history = train(
                            model, 
                            model_file_name,
                            train_loader, 
                            val_loader, 
                            lr=lr,
                            num_epochs=epochs,
                            weight_decay=weight_decay,
                            device=device,
                            continue_training=False
                        )
                        
                        plot_results(epochs, train_loss_history, train_acc_history, val_loss_history, val_acc_history)
                        
                        avg_test_loss, avg_test_acc, confusion_matrix_data = evaluate(model, test_loader, device=device)
                         
                        # create dictionary storing all results and save to results.json
                        results[model_file_name] = {
                            'lr': lr,
                            'batch_size': batch_size,
                            'epochs': epochs,
                            'weight_decay': weight_decay,
                            'dropout': dropout,
                            'train_loss_history': train_loss_history,
                            'train_acc_history': train_acc_history,
                            'val_loss_history': val_loss_history,
                            'val_acc_history': val_acc_history,
                            'avg_test_loss': avg_test_loss,
                            'avg_test_acc': avg_test_acc,
                            'confusion_matrix_data': confusion_matrix_data
                        }
                        
                        with open ("plots/results.json", "w") as f:
                            json.dump(results, f, indent=2)


if __name__ == '__main__':
    grid_search()