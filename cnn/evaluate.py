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

import json
import sys
import os
import argparse

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
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images).squeeze()
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            test_acc += (predicted == labels).sum().item()
    
    avg_test_loss = test_loss / len(data_loader.dataset)
    avg_test_acc = test_acc / len(data_loader.dataset)

    
    return avg_test_loss, avg_test_acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    
    # Add arguments
    parser.add_argument('--eval_whole_set', action='store_true', help='Evaluate the whole dataset')
    parser.add_argument('--eval_subclasses', action='store_true', help='Evaluate subclasses')

    # Parse the arguments
    args = parser.parse_args()
    eval_whole_set = args.eval_whole_set
    eval_subclasses = args.eval_subclasses
    
    if not eval_whole_set and not eval_subclasses:
        eval_whole_set = True

    
    BATCH_SIZE = 16

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if eval_whole_set:
        test_ds = ImageFolder(root='data/test', transform=data_transforms)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        model = BinaryClassifier()
        model.load_state_dict(torch.load('model.pth', map_location='cuda:0' if torch.cuda.is_available() else 'cpu'))
        model.to('cuda')  # Move the model to GPU
        model.eval()
        avg_test_loss, avg_test_acc = evaluate(model, test_loader, device='cuda')
        
        print(f'Test loss: {avg_test_loss:.4f}')
        print(f'Test accuracy: {avg_test_acc:.4f}')
        
        



    if eval_subclasses:
        subclasses = os.listdir('data/test/fake')
        subclass_results = {}
        
        for subclass in subclasses:
            # delete temp folder if it exists
            if os.path.exists('temp'):
                os.system('rm -rf temp')
                
            # create folder called temp and copy the real and fake images from data/test/real/{subclass} and data/test/fake/{subclass} into it
            os.makedirs('temp/subclass_real', exist_ok=True)
            os.makedirs('temp/subclass_fake', exist_ok=True)
            
            # copy real from data/test/real/{subclass} to temp/subclass_real
            print(f'Copying real images from data/test/real/{subclass} to temp/subclass_real')
            real_images = os.listdir(f'data/test/real/{subclass}')
            for image in real_images:
                os.system(f'cp data/test/real/{subclass}/{image} temp/subclass_real')
            
            print(f'Copying fake images from data/test/fake/{subclass} to temp/subclass_fake')
            # copy fake from data/test/fake/{subclass} to temp/subclass_fake
            fake_images = os.listdir(f'data/test/fake/{subclass}')
            for image in fake_images:
                os.system(f'cp data/test/fake/{subclass}/{image} temp/subclass_fake')
                
            # create a new ImageFolder for the subclass
            subclass_test_ds = ImageFolder(root=f'temp', transform=data_transforms)
            
            stop = input('stop')
            
            # subclass_test_ds = ImageFolder(root=f'data/test/{subclass}', transform=data_transforms)
            subclass_test_loader = DataLoader(subclass_test_ds, batch_size=BATCH_SIZE, shuffle=False)
            
            model = BinaryClassifier()
            model.load_state_dict(torch.load('model.pth'))
            model.to('cuda')
            model.eval()
            
            avg_test_loss, avg_test_acc = evaluate(model, subclass_test_loader, device='cuda')
            subclass_results[subclass] = {'loss': avg_test_loss, 'acc': avg_test_acc}
            
            os.system('rm -rf temp')
        
        for subclass in subclass_results:
            print(f'{subclass}:')
            print(f'Loss: {subclass_results[subclass]["loss"]:.4f}')
            print(f'Accuracy: {subclass_results[subclass]["acc"]:.4f}')
            print()
        
        # save results to a json file
        with open('subclass_results.json', 'w') as f:
            json.dump(subclass_results, f)        
        
        