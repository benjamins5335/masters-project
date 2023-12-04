import json

if __name__ == '__main__':
    data = json.load(open('results.json'))
    
    # Create a list of tuples where each tuple contains the key and final val_acc
    val_acc_list = [(key, float(data[key]["val_acc_history"][-1])) for key in data]

    # Sort the list of tuples based on the final val_acc in descending order
    val_acc_list = sorted(val_acc_list, key=lambda x: x[1], reverse=True)
    
    for key, val_acc in val_acc_list:
        print(key, val_acc)
    
    # for every key in data
    # remove 'avg_test_loss', 'avg_test_acc', 'confusion_matrix_data', 'accuracy' if present
    # for key in data:
    #     if 'avg_test_loss' in data[key]:
    #         del data[key]['avg_test_loss']
    #     if 'avg_test_acc' in data[key]:
    #         del data[key]['avg_test_acc']
    #     if 'confusion_matrix_data' in data[key]:
    #         del data[key]['confusion_matrix_data']
    #     if 'accuracy' in data[key]:
    #         del data[key]['accuracy']
        
    # # save the new data
    # with open ("results.json", "w") as f:
    #     json.dump(data, f, indent=2)