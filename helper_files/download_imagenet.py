import requests
import json

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# def download_image(url, file_name):
    