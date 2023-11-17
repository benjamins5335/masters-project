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
