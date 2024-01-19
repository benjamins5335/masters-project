import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torch.utils.data import DataLoader

from models.binary_classifier import BinaryClassifier

import json
import os
import argparse
import plotly.figure_factory as ff


def evaluate(model, data_loader, device='cuda'):
    """Evaluate the model on various datasets.

    Args:
        model (pth model): The model to evaluate.
        data_loader (DataLoader): The data loader to use.
        device (str, optional): Specify CPU if necessary. Defaults to 'cuda'.

    Returns:
        avg_test_loss (float): The average test loss.
        avg_test_acc (float): The average test accuracy.
        confusion_matrix_data (2d-array): The confusion matrix data.
    """

    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    test_loss = 0
    test_acc = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images).squeeze()
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
            
            test_acc += (predicted == labels).sum().item()
    
    confusion_matrix_data = [
        [true_positives, false_positives],
        [false_negatives, true_negatives]
    ]
    

    avg_test_loss = test_loss / len(data_loader.dataset)
    avg_test_acc = test_acc / len(data_loader.dataset)

    
    return avg_test_loss, avg_test_acc, confusion_matrix_data


def write_confusion_matrix_data_to_csv(confusion_matrix_data, name):
    """Write the confusion matrix data to a csv file.

    Args:
        confusion_matrix_data (2d-array): The confusion matrix data.
    """
    
    csv_file_path = 'plots/confusion_matrix_data.csv'
    
    # if the file doesn't exist, create it and write the header
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w') as f:
            f.write("Subclass,True Positives,False Positives,False Negatives,True Negatives\n")
    
    
    with open(csv_file_path, 'a') as f:
        f.write('{},{},{},{},{}\n'.format(name, confusion_matrix_data[0][0], confusion_matrix_data[0][1], confusion_matrix_data[1][0], confusion_matrix_data[1][1]))


def create_confusion_matrix(results, name):
    """Create a confusion matrix from the results.

    Args:
        results (2d-array): The results from the evaluate function.
        name (str): Description of the confusion matrix (e.g., 'whole_set', 'unseen', 'dog', 'cat', etc.)
    """

    # Restrcuture the results to be in the correct format for the matrix
    classes = ['Real', 'Fake']
    x = classes
    y = classes[::-1]
    z = results[::-1]
    
    # Convert all values to strings
    z_text = [[str(y) for y in x] for x in z]
    
    
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='gray')  # Change colorscale to 'gray'
    fig.update_layout(title='Confusion Matrix', xaxis_title='Actual', yaxis_title='Predicted')
    fig.write_image(f'plots/{name}_confusion_matrix.png')


if __name__ == '__main__':
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    
    # Add arguments
    parser.add_argument('--whole_set', action='store_true', help='Evaluate the whole dataset')
    parser.add_argument('--subclasses', action='store_true', help='Evaluate subclasses')
    parser.add_argument('--unseen', action='store_true', help='Evaluate unseen classes')
    parser.add_argument('--model_path', type=str, help='Path to the model')

    # Parse the arguments
    args = parser.parse_args()
    eval_whole_set = args.whole_set
    eval_subclasses = args.subclasses
    eval_unseen = args.unseen
    model_path = args.model_path
    
    # try pull batch size from model path and default to 32 if it doesn't exist
    try:
        batch_size = int(model_path.split('_')[len(model_path.split('_')) - 4])
    except Exception:
        batch_size = 32

    # run whole set evaluation if no arguments are passed
    if not eval_whole_set and not eval_subclasses and not eval_unseen:
        eval_whole_set = True

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # load resnet model if the model path contains resnet18
    # use custom model otherwise
    if "resnet" in model_path:
        pretrained_model = True
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)  
    else:
        pretrained_model = False
        model = BinaryClassifier()
        
    model.load_state_dict(torch.load(model_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)  
    model.eval()
    

    if eval_whole_set:
        test_ds = ImageFolder(root='data/test', transform=data_transforms)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        avg_test_loss, avg_test_acc, confusion_matrix_data = evaluate(model, test_loader, device)
        
        print(f'Test loss: {avg_test_loss:.4f}')
        print(f'Test accuracy: {avg_test_acc:.4f}')
        
        create_confusion_matrix(confusion_matrix_data, 'whole_set')
        
    
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
            subclass_test_loader = DataLoader(subclass_test_ds, batch_size=batch_size, shuffle=False)

            avg_test_loss, avg_test_acc, confusion_matrix_data = evaluate(model, subclass_test_loader, device)
            create_confusion_matrix(confusion_matrix_data, subclass)
            
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
        
    if eval_unseen:
        test_ds = ImageFolder(root='unseen', transform=data_transforms)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        avg_test_loss, avg_test_acc, confusion_matrix_data = evaluate(model, test_loader, device='cuda')
        
        print(f'Test loss: {avg_test_loss:.4f}')
        print(f'Test accuracy: {avg_test_acc:.4f}')
        
        create_confusion_matrix(confusion_matrix_data, 'unseen')
        