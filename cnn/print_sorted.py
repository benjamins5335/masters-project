import json

if __name__ == '__main__':
    data = json.load(open('results.json'))
    
    # Create a list of tuples where each tuple contains the key and accuracy
    accuracy_list = [(key, float(data[key]["accuracy"])) for key in data]

    # Sort the list of tuples based on accuracy in descending order
    accuracy_list = sorted(accuracy_list, key=lambda x: x[1], reverse=True)
    
    for key, accuracy in accuracy_list:
        print(key, accuracy)
        
    
