import cv2
import os

base_url = 'https://image-net.org/data/winter21_whole/{}.tar'
base_path = '../data/real'
 

def process_images():
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.JPEG'):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                
                height, width = img.shape[:2]
                if width < 96 or height < 96:
                    os.remove(file_path)
                    
                elif width > 96 or height > 96:
                    left = int((width - 96)/2)
                    top = int((height - 96)/2)
                    right = left + 96
                    bottom = top + 96
                    
                    img = img[top:bottom, left:right]
                    
                    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
                    
                    cv2.imwrite(file_path, img)
    
                    
                    
                    
                    