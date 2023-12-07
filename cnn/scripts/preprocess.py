import random
import cv2
import os


def downsample_and_save(file_path, img, width, height):
    """Deterministically downsamples the image to 96x96 and saves it to the given file path

    Args:
        file_path (str): Path to save the image to
        img (image): Image to downsample
        width (int): Width of the image
        height (int): Height of the image
    """
    crop_size = min(width, height) # dimensions of the smallest side
    
    # crop the image
    left = (width - crop_size) // 2
    right = left + crop_size
    top = (height - crop_size) // 2
    bottom = top + crop_size
    img = img[top:bottom, left:right]
    
    # downsample and save
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR) 
    cv2.imwrite(file_path, img) 
    
    
    
def write_1000_fake(input_dir, output_dir):
    """Writes 1000 fake images to the given output directory

    Args:
        input_dir (str): Path to the directory containing the images to be downsampled
        output_dir (str): Path to the directory to save the downsampled images to
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    print(f'Processing images in {input_dir}...')
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # get current dir
            subclass = root.split('/')[-1]

            # create dir if it doesn't exist
            if not os.path.exists(os.path.join(output_dir, subclass)):
                os.mkdir(os.path.join(output_dir, subclass))
            
            # only images
            if file.endswith('.JPEG') or file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, subclass, file)
                
                img = cv2.imread(input_file_path)
                
                height, width = img.shape[:2]
                if width >= 96 or height >= 96: # check if image is already 96x96
                    downsample_and_save(output_file_path, img, width, height)
                    print(f'Saved {output_file_path}')


def choose_1000_real(base_path):
    """Chooses 1000 real images from each class

    Args:
        base_path (str): Path to the directory containing the real images

    Returns:
        dict: Dictionary containing the number of images to download for each subclass
    """
    print('Choosing 1000 real images...')
    
    number_to_download = {}
    for root, dirs, files in os.walk(base_path + '/real'):
        for dir in dirs: # for each class: cat, dog, etc.
            print(f'Processing {dir}...')
            real_images = {}
            total_deficit = 0

            # iterate through every file in dir
            files = os.listdir(os.path.join(root, dir))
            
            # count number of images for each synset_id
            for file in files:
                if file.lower().endswith(('.jpeg', '.jpg', '.png', '.JPEG')):
                    synset_id = file.split('_')[0]
                    if synset_id not in real_images:
                        real_images[synset_id] = 1
                    else:
                        real_images[synset_id] += 1
                
                
            # check if all values are greater than 1000 in real_images
            # if not, the deficit is calculated
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
                
            # images are 'borrowed' from other classes to make up for the deficit
            for key, value in real_images.items():
                if value - 1000 > total_deficit and total_deficit > 0:
                    number_to_download_class[key] = 1000 + total_deficit
                    total_deficit = 0
                
            number_to_download.update(
                {
                    dir: number_to_download_class
                }
            )
                
    return number_to_download

def write_1000_real(input_dir, output_dir, number_to_download):
    """Writes 1000 real images to the given output directory

    Args:
        input_dir (str): Path to the directory containing the images to be downsampled
        output_dir (str): Path to the directory to save the downsampled images to
        number_to_download (dict): Dictionary containing the number of images to download for each subclass
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    # iterate through every file in dir
    for root, dirs, files in os.walk(input_dir): # data_raw/real
        for dir in dirs: # cat, dog, etc.
            if not os.path.exists(os.path.join(output_dir, dir)): # data/real/{cat, dog, etc.}
                os.mkdir(os.path.join(output_dir, dir))
            number_to_download_class = number_to_download[dir]
            
            files = os.listdir(os.path.join(input_dir, dir)) 
            for file in files:
                if file.lower().endswith(('.jpeg', '.jpg', '.png', '.JPEG')):
                    input_file_path = os.path.join(input_dir, dir, file)
                    output_file_path = os.path.join(output_dir, dir, file)                      
                    
                    synset_id = file.split('_')[0]
                    if number_to_download_class[synset_id] > 0:
                        img = cv2.imread(input_file_path)
                        height, width = img.shape[:2]
                        if width >= 96 or height >= 96:
                            number_to_download_class[synset_id] -= 1
                            downsample_and_save(output_file_path, img, width, height)
                            print(f'Saved {output_file_path}')
                        else:
                            print(f'{input_file_path} is too small. Skipping...')
                            
                            
def separate_into_train_and_test(split_percentage):
    """Helper function to separate the images into train and test

    Args:
        split_percentage (float): Percentage of images to be used for training.
            The rest will be used for testing.
    """
    print('Separating into train and test...')
    for root, dirs, files in os.walk('../data/real'):
        for dir in dirs:
            print(f'Processing {dir}...')
            if not os.path.exists(os.path.join('../data', 'train', 'real', dir)):
                os.makedirs(os.path.join('../data', 'train', 'real', dir), exist_ok=True)
            if not os.path.exists(os.path.join('../data', 'test', 'real', dir)):
                os.makedirs(os.path.join('../data', 'test', 'real', dir), exist_ok=True)
        
                
            files = os.listdir(os.path.join(root, dir))
            
            random.shuffle(files)
            
            train_size = int(split_percentage * len(files))
            test_size = len(files) - train_size
            
            count = 0
            for file in files: 
                if count < split_percentage * len(files):
                    mode = 'train'
                else:
                    mode = 'test'
                        
                old_path = os.path.join(root, dir, file)
                new_path = os.path.join("../data", mode, "real", dir, file)
                
                os.rename(old_path, new_path)
                print(f'Moved {new_path}')
                count += 1
                

    # do the same for fake images
    for root, dirs, files in os.walk('../data/fake'):
        for dir in dirs:
            print(f'Processing {dir}...')
            if not os.path.exists(os.path.join('../data', 'train', 'fake', dir)):
                os.makedirs(os.path.join('../data', 'train', 'fake', dir), exist_ok=True)
            if not os.path.exists(os.path.join('../data', 'test', 'fake', dir)):
                os.makedirs(os.path.join('../data', 'test', 'fake', dir), exist_ok=True)
        
                
            files = os.listdir(os.path.join(root, dir))
            
            random.shuffle(files)
            
            train_size = int(split_percentage * len(files))
            test_size = len(files) - train_size
            
            count = 0
            for file in files: 
                if count < split_percentage * len(files):
                    mode = 'train'
                else:
                    mode = 'test'
                        
                old_path = os.path.join(root, dir, file)
                new_path = os.path.join("../data", mode, "fake", dir, file)
                
            
                os.rename(old_path, new_path)
                print(f'Moved {new_path}')
                count += 1
            
            
if __name__ == "__main__":
    print('Starting preprocessing...')
    
    input_dir = 'data_raw'
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    # write_1000_real(input_dir + "/real", output_dir + "/real", number_to_download)
    # write_1000_fake(input_dir + "/fake", output_dir + "/fake")
    # separate_into_train_and_test(0.8)
    
    

