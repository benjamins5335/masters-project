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
    for superclass in os.listdir(input_dir):
        if superclass.startswith("."):
            continue
        in_fake = os.path.join(input_dir, superclass, "fake")
        out_fake = os.path.join(output_dir, superclass, "fake")
        os.makedirs(out_fake, exist_ok=True)

        for file in os.listdir(in_fake):
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                img = cv2.imread(os.path.join(in_fake, file))
                if img is None:
                    continue
                h, w = img.shape[:2]
                downsample_and_save(os.path.join(out_fake, file), img, w, h)


def choose_1000_real(base_path):
    print("Choosing 1000 real images...")

    number_to_download = {}

    # Loop over superclass directories: fish, rodent...
    for superclass in os.listdir(base_path):
        if superclass.startswith("."):
            continue
        real_dir = os.path.join(base_path, superclass, "real")
        if not os.path.isdir(real_dir):
            continue
        
        print(f"Processing {superclass}...")
        real_images = {}
        total_deficit = 0

        files = os.listdir(real_dir)

        # Count images per synset
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                synset = file.split("_")[0]
                real_images[synset] = real_images.get(synset, 0) + 1

        # Decide how many to pick
        number_to_download_class = {}
        for key, value in real_images.items():
            number_to_download_class[key] = min(200, value)

        number_to_download[superclass] = number_to_download_class

    return number_to_download

def write_1000_real(input_dir, output_dir, number_to_download):

    for superclass, synset_counts in number_to_download.items():
        if superclass.startswith("."):
            continue
        in_real = os.path.join(input_dir, superclass, "real")
        out_real = os.path.join(output_dir, superclass, "real")

        os.makedirs(out_real, exist_ok=True)

        for file in os.listdir(in_real):
            if not file.lower().endswith(('.jpeg', '.jpg', '.png')):
                continue

            synset = file.split("_")[0]
            if synset_counts.get(synset, 0) <= 0:
                continue

            img = cv2.imread(os.path.join(in_real, file))
            if img is None:
                continue

            h, w = img.shape[:2]
            if h < 96 or w < 96:
                print(f"Skipping too small: {file}")
                continue

            synset_counts[synset] -= 1
            downsample_and_save(os.path.join(out_real, file), img, w, h)

                            
def separate_into_train_and_test(split):
    for superclass in os.listdir("unseen"):
        if superclass.startswith("."):
            continue
        for mode in ["real", "fake"]:
            src = os.path.join("unseen", superclass, mode)
            if not os.path.isdir(src):
                continue

            dst_train = os.path.join("data/train", superclass, mode)
            dst_test  = os.path.join("data/test", superclass, mode)
            os.makedirs(dst_train, exist_ok=True)
            os.makedirs(dst_test, exist_ok=True)

            files = os.listdir(src)
            random.shuffle(files)

            cutoff = int(len(files) * split)

            for i, file in enumerate(files):
                dst = dst_train if i < cutoff else dst_test
                os.rename(os.path.join(src, file), os.path.join(dst, file))
            
if __name__ == "__main__":
    print('Starting preprocessing...')
    
    input_dir = 'unseen_raw'
    output_dir = 'unseen'
    os.makedirs(output_dir, exist_ok=True)
    
    number_to_download = choose_1000_real(input_dir)
    write_1000_real(input_dir, output_dir, number_to_download)
    write_1000_fake(input_dir, output_dir)
    separate_into_train_and_test(0)
    
    

