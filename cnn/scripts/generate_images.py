import json
from pathlib import Path
import random
import sys
from diffusers import StableDiffusionXLPipeline
import torch
import argparse


def generate_image(pipe, prompt):
    """Generates an image based on the given prompt

    Args:
        pipe (pipe): Pipeline object used for generating images
        prompt (str): Prompt used for generating the image

    Returns:
        image: Image generated by the pipeline
    """
    # negative prompts used for every file
    negative_prompt = "ugly, deformed, animation, anime, cartoon, comic, cropped, out of frame, low res"
    

    image = pipe(
        prompt=prompt,
        num_inference_steps=40,
        negative_prompt=negative_prompt   
    ).images[0]
        
    return image 


def generate_all_images():   
    """Function to generate all images outlined by the subclasses in image_classes.json
    """
    all_prompts = json.load(open("scripts/image_classes.json"))

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")

    pipe.enable_xformers_memory_efficient_attention() # improves performance

    images_per_subclass = 5000
    
    for class_obj in all_prompts: # for each class: dog, cat, etc.
        count = 0
        image_class = class_obj["class"]
        image_prompt = class_obj["prompt"]
        variations = class_obj["variations"] # used to add diversity to dataset
        subclasses = [item['description'] for item in class_obj['wnids']] # list of subclass names
        
        # create folder for class in ../data/fake if it doesn't exist
        class_folder = Path(f"data_raw/fake/{image_class}")
        class_folder.mkdir(parents=True, exist_ok=True)
    
        for subclass in subclasses:
            for _ in range(images_per_subclass):
                prompt = image_prompt.format(subclass) + ", " + random.choice(variations)
                image_filename = f"{image_class}_{count}.JPEG"
                image_path = class_folder / image_filename
                
                # allows script to pick up where it left off if it crashes
                if not image_path.exists():
                    image = generate_image(pipe, prompt)
                    # save to fake dataset found in ../data/fake/{name of class} as JPEG      
                    image.save(image_path)
                    print(f"Saved {image_filename} to {image_path}.")
                else:
                    print(f"Skipping {image_filename} as it already exists.")
                
                count += 1
                

def generate_image_from_class(chosen_class):
    """Generates images for a specified class

    Args:
        chosen_class (str): Class name for image generation (e.g., 'dog', 'cat').

    Raises:
        ValueError: If the chosen class is not in the list of valid classes
    """
    all_prompts = json.load(open("scripts/image_classes.json"))
    valid_classes = [item['class'] for item in all_prompts]
    if chosen_class not in valid_classes:
        raise ValueError(f"Invalid class name. Valid class names are: {valid_classes}")
    
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()
    
    images_per_subclass = 5000
    total_images = -1
    
    for class_obj in all_prompts:

        if chosen_class == class_obj["class"]:

            image_prompt = class_obj["prompt"]
            variations = class_obj["variations"]

            subclasses = [item['description'] for item in class_obj['wnids']]
            total_subclasses = len(subclasses)
            total_images = total_subclasses * images_per_subclass

            class_folder = Path(f"data_raw/fake/{chosen_class}")
            class_folder.mkdir(parents=True, exist_ok=True)


            for count in range(total_images):
                subclass_index = count // images_per_subclass
                subclass = subclasses[subclass_index]
                prompt = image_prompt.format(subclass) + ", " + random.choice(variations)
                image_filename = f"{chosen_class}_{count}.JPEG"
                image_path = class_folder / image_filename

                if not image_path.exists():
                    image = generate_image(pipe, prompt)
                    # save to fake dataset found in ../data/fake/{name of class} as JPEG      
                    image.save(image_path)
                    print(f"Saved {image_filename} to {image_path}.")
                else:
                    print(f"Skipping {image_filename} as it already exists.")

                count += 1


if __name__ == "__main__":

    if len(sys.argv) == 1:
        generate_all_images()
    else:
        class_name = sys.argv[1]

        generate_image_from_class(class_name)
