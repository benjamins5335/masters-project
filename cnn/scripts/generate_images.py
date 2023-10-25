import json
from pathlib import Path
import random
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image



def generate_image(pipe, prompt):
    negative_prompt = "ugly, deformed, animation, anime, cartoon, comic, cropped, out of frame, low res"
    
    
    

    image = pipe(
        prompt=prompt,
        num_inference_steps=40,
        negative_prompt=negative_prompt
    
    ).images[0]
        
    return image 


def generate_all_images():   
    all_prompts = json.load(open("imagenet_classes.json"))
    print(all_prompts)
    
    stop = input("Enter 'y' to stop entering prompts: ")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")
    

    pipe.enable_xformers_memory_efficient_attention()

    images_per_subclass = 5
    
    
    for class_obj in all_prompts:
        count = 0
        image_class = class_obj["class"]
        image_prompt = class_obj["prompt"]
        variations = class_obj["variations"]
        subclasses = [item['description'] for item in class_obj['wnids']]
        
        # create folder for class in ../data/fake if it doesn't exist
        Path(f"../data/fake/{image_class}").mkdir(parents=True, exist_ok=True)
        

        for subclass in subclasses:
            for i in range(images_per_subclass):
                prompt = image_prompt.format(subclass) + ", " + random.choice(variations)
                image = generate_image(pipe, prompt)
                
                # save to fake dataset found in ../data/fake/{name of class} as JPEG      
                image.save(f"../data/fake/{image_class}/{image_class}_{count}.JPEG")
                count += 1
                
def test_function():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")
    

    pipe.enable_xformers_memory_efficient_attention()
    
    image = generate_image(pipe, "dirty stinking hippie from LA long hair and really baggy clothes smoking bongs and playing hacky sack")
    image.save("test.jpeg")


if __name__ == "__main__":
    generate_all_images()
    
    
