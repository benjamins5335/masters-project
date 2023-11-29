from train import plot_results
from evaluate import create_confusion_matrix
import json


def get_results(model_path):
    with open('results.json', 'r') as f:
        results = json.load(f)
    
    return results[model_path]
    
        
if __name__ == "__main__":
    results = get_results("0.001_64_30_0.001_0.4")
    n_epochs = results['epochs']
    train_loss_history = results['train_loss_history']
    train_acc_history = results['train_acc_history']
    val_loss_history = results['val_loss_history']
    val_acc_history = results['val_acc_history']
    
    plot_results(n_epochs, train_loss_history, train_acc_history, val_loss_history, val_acc_history)
    