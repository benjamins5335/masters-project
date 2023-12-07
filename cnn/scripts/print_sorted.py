import json


def print_sorted():
    """Prints the results in results.json in descending order of the final validation accuracy
    """
    data = json.load(open('results.json'))
    
    # Create a list of tuples where each tuple contains the key and final val_acc
    val_acc_list = [(key, float(data[key]["val_acc_history"][-1])) for key in data]

    # Sort the list of tuples based on the final val_acc in descending order
    val_acc_list = sorted(val_acc_list, key=lambda x: x[1], reverse=True)
    
    for key, val_acc in val_acc_list:
        print(key, val_acc)

if __name__ == '__main__':
    print_sorted()