import json
from pathlib import Path
import random
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import argparse


def generate_image(pipe, prompt):
    negative_prompt = "ugly, deformed, animation, anime, cartoon, comic, cropped, out of frame, low res"
    

    image = pipe(
        prompt=prompt,
        num_inference_steps=40,
        negative_prompt=negative_prompt
        # num_images_per_prompt=2
    
    ).images[0]
        
    return image 


def generate_all_images():   
    all_prompts = json.load(open("image_classes.json"))
        
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()

    images_per_subclass = 1000
    
    for class_obj in all_prompts:
        image_class = class_obj["class"]
        image_prompt = class_obj["prompt"]
        variations = class_obj["variations"]
        
        subclasses = [item['description'] for item in class_obj['wnids']]
        total_subclasses = len(subclasses)
        total_images = total_subclasses * images_per_subclass
        
        class_folder = Path(f"data/fake/{image_class}")
        class_folder.mkdir(parents=True, exist_ok=True)
        
    
        for count in range(total_images):
            subclass_index = count // images_per_subclass
            subclass = subclasses[subclass_index]
            prompt = image_prompt.format(subclass) + ", " + random.choice(variations)
            image_filename = f"{image_class}_{count}.JPEG"
            image_path = class_folder / image_filename
            
            if not image_path.exists():
                image = generate_image(pipe, prompt)
                # save to fake dataset found in ../data/fake/{name of class} as JPEG      
                image.save(image_path)
                print(f"Saved {image_filename} to {image_path}.")
            else:
                print(f"Skipping {image_filename} as it already exists.")
            
            count += 1
            



def generate_image_from_class(chosen_class):
    all_prompts = json.load(open("image_classes.json"))
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
    
    images_per_subclass = 1000
    total_images = -1
    
    for class_obj in all_prompts:

        if chosen_class == class_obj["class"]:

            image_prompt = class_obj["prompt"]
            variations = class_obj["variations"]
            
            subclasses = [item['description'] for item in class_obj['wnids']]
            total_subclasses = len(subclasses)
            total_images = total_subclasses * images_per_subclass
            
            class_folder = Path(f"data/fake/{chosen_class}")
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
    # Set up command-line argument parsing
    # parser = argparse.ArgumentParser(description="Generate images based on the specified class.")
    # parser.add_argument("class_name", type=str, help="Class name for image generation (e.g., 'dog', 'cat').")
    
    # # Parse command-line arguments
    # args = parser.parse_args()

    try:
        # Use the provided class name
        generate_all_images()
    except ValueError as e:
        print(e)
    
