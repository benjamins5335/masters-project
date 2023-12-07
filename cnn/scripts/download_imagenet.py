import tarfile
import requests
import json
import cv2
import os

BASE_URL = 'https://image-net.org/data/winter21_whole/{}.tar'
BASE_PATH = '../data/real'


def read_json(json_file):
    """Reads a json file and returns the data
    
    Args:
        json_file (str): path to json file
        
    Returns:
        data (dict): data from json file
    """
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def download_wnid(wnid, class_folder):    
    """Downloads the tar file for the given wnid and extracts it to the given class folder

    Args:
        wnid (str): The identifier used by ImageNet to identify a class
        class_folder (str): Path to the folder where the tar file will be downloaded and extracted
    """
    print("Downloading '{}' for class '{}'".format(wnid, class_folder))
    
    url = BASE_URL.format(wnid)
    r = requests.get(url, allow_redirects=True, stream=True)
    
    tar_file_path = os.path.join(class_folder, wnid + '.tar') 
    with open(tar_file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    
    # Extract and delete the tar file
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(path=class_folder)
    os.remove(tar_file_path)


def download_all():
    """Function to download all synsets outlined in imagenet_classes.json
    """
    data = read_json('imagenet_classes.json')
    if not os.path.exists(BASE_PATH):
        os.mkdir(BASE_PATH)
        
    
    for item in data:
        class_name = item['class']
        class_folder = os.path.join(BASE_PATH, class_name)
        wnids = item['wnids']
        
        if not os.path.exists(class_folder):
            os.mkdir(class_folder)
            
        for wnid in wnids:
            download_wnid(wnid, class_folder)

    print("Done")
               

if __name__ == '__main__':
    download_all()
