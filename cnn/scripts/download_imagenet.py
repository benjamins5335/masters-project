import tarfile
import requests
import json
import cv2
import os

base_url = 'https://image-net.org/data/winter21_whole/{}.tar'
base_path = '../data/real'

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def download_wnid(wnid, class_folder):
    
    print("Downloading '{}' for class '{}'".format(wnid, class_folder))
    
    url = base_url.format(wnid)
    
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
    data = read_json('imagenet_classes.json')
    if not os.path.exists(base_path):
        os.mkdir(base_path)
        
    for item in data:
        class_name = item['class']
        class_folder = os.path.join(base_path, class_name)
        wnids = item['wnids']
        
        if not os.path.exists(class_folder):
            os.mkdir(class_folder)
            
        for wnid in wnids:
            download_wnid(wnid, class_folder)

    print("Done")
               





if __name__ == '__main__':
    # download_all()
    download_wnid('n02129604', '../data/real/cat')