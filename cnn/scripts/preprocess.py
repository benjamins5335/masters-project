import cv2
import os
import argparse
import time
from collections import defaultdict

def downsample(base_path):
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
               


def choose_1000_real(base_path):
    print('Choosing 1000 real images...')
    
    number_to_download = {}
    for root, dirs, files in os.walk(base_path + '/real'):
        for dir in dirs:
            print(f'Processing {dir}...')
            real_images = {}
            total_deficit = 0

            # iterate through every file in dir
            files = os.listdir(os.path.join(root, dir))
            for file in files:
                # print(f'Processing {file}...')
                if file.lower().endswith(('.jpeg', '.jpg', '.png', '.JPEG')):
                    synset_id = file.split('_')[0]
                    if synset_id not in real_images:
                        real_images[synset_id] = 1
                    else:
                        real_images[synset_id] += 1
                
            # check if all values are greater than 1000 in real_images
            if all(value > 1000 for value in real_images.values()):
                number_to_download_class = {}
                for key, value in real_images.items():
                    number_to_download_class[key] = 1000
            else:
                number_to_download_class = {}
                for key, value in real_images.items():
                    if value > 1000:
                        number_to_download_class[key] = 1000
                    else:
                        number_to_download_class[key] = value
                        total_deficit += 1000 - value
                
            
            for key, value in real_images.items():
                if value - 1000 > total_deficit and total_deficit > 0:
                    number_to_download_class[key] = 1000 + total_deficit
                    total_deficit = 0
                
            number_to_download.update(
                {
                    dir: number_to_download_class
                }
            )
            
    print(number_to_download)
    
    return number_to_download

def write_1000_real(base_path, number_to_download):
    output_path = base_path + '/real_new'
    # make directory for output
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # iterate through every file in dir
    for root, dirs, files in os.walk(base_path + '/real'):
        for dir in dirs:
            # make dir for class
            if not os.path.exists(os.path.join(output_path, dir)):
                os.mkdir(os.path.join(output_path, dir))
            number_to_download_class = number_to_download[dir]
            # iterate through every file in dir
            files = os.listdir(os.path.join(root, dir))
            for file in files:
                # print(f'Processing {file}...')
                if file.lower().endswith(('.jpeg', '.jpg', '.png', '.JPEG')):
                    synset_id = file.split('_')[0]
                    if number_to_download_class[synset_id] > 0:
                        file_path = os.path.join(root, dir, file)
                        img = cv2.imread(file_path)
                        height, width = img.shape[:2]
                        if width < 96 or height < 96:
                            os.remove(file_path)
                            
                        elif width > 96 or height > 96:
                            number_to_download_class[synset_id] -= 1

                            crop_size = min(width, height)
                            left = (width - crop_size) // 2
                            right = left + crop_size
                            top = (height - crop_size) // 2
                            bottom = top + crop_size
                            
                            img = img[top:bottom, left:right]
                            
                            img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
                            cv2.imwrite(os.path.join(output_path, dir, file), img)
        
                  
if __name__ == "__main__":
    print('Starting preprocessing...')
    parser = argparse.ArgumentParser(description='Process images in a directory.')
    parser.add_argument('base_path', type=str, help='Path to the base directory containing images to process.', default='../data')
    args = parser.parse_args()
    
    path = args.base_path
    
    # number_to_download = choose_1000_real(path)
    # write_1000_real(path, number_to_download)
    downsample(path)
    
    
    

