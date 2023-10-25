import cv2
import os
import argparse
import time

def process_images(base_path):
    print(f'Processing images in {base_path}...')
    
    for root, dirs, files in os.walk(base_path):
        
        for file in files:
            
            if file.endswith('.JPEG') or file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                print(f'Processing {file}...')
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                
                height, width = img.shape[:2]
                if width < 96 or height < 96:
                    os.remove(file_path)
                    
                elif width > 96 or height > 96:
                    crop_size = min(width, height)
                    left = (width - crop_size) // 2
                    right = left + crop_size
                    top = (height - crop_size) // 2
                    bottom = top + crop_size
                    
                    img = img[top:bottom, left:right]
                    
                    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
                    
                    cv2.imwrite(file_path, img)
               
                    
if __name__ == "__main__":
    print('Starting preprocessing...')
    parser = argparse.ArgumentParser(description='Process images in a directory.')
    parser.add_argument('base_path', type=str, help='Path to the base directory containing images to process.')
    args = parser.parse_args()

    process_images(args.base_path)