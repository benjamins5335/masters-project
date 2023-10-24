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
    all_prompts = [
        {
            "class": "dog",
            "prompt": "A photo of a {}, real-life setting.",
            "subclasses": [
                "labrador retriever dog",
                "german shepherd dog",
                "siberian husky dog",
                "beagle dog",
                "dalmatian dog"
            ],
            "variations": [
                "running","jumping","playing","sleeping","eating","inside","outside"
            ]
            
        },
        {
            "class": "cat",
            "prompt": "A photo of a {}, real-life setting.",
            "subclasses": [
                "persian cat",
                "siamese cat",
                "tabby cat",
                "egyptian mau cat",
                "tiger cat"
            ],
            "variations": [
                "lying down","standing","sleeping","eating","playing","inside","outside"
            ]
            
            
        },
        {
            "class": "car",
            "prompt": "A photo of a {}, real-life setting.",
            "subclasses": [
                "passenger car",
                "estate car",
                "taxi",
                "convertible car",
                "jeep"
            ],
            "variations": [
                "parked","driving","inside","outside","front","back"
            ]
        },
        {
            "class": "ship",
            "prompt": "A photo of a {}, real-life setting.",
            "subclasses": [
                "fireboat",
                "speedboat",
                "container ship",
                "ocean liner",
                "aircraft carrier"
            ],
            "variations": [
                "docked","sailing","outside","front","back"
            ]
        },
        {
            "class": "bird",
            "prompt": "A photo of a {}, real-life setting.",
            "subclasses": [
                "robin",
                "goldfinch",
                "jay bird",
                "magpie",
                "bald eagle"
            ],
            "variations": [
                "flying","perched","eating","outside","in nest"
            ]
        },
        {
            "class": "primate",
            "prompt": "A photo of a {}, real-life setting.",
            "subclasses": [
                "orangutan",
                "chimpanzee",
                "gibbon",
                "gorilla",
                "spider monkey"
            ],
            "variations": [
                "lying down","standing","sleeping","eating","playing","inside","outside"
            ]
        },
        {
            "class": "spider",
            "prompt": "A photo of a {}, real-life setting.",
            "subclasses": [
                "black and gold garden spider",
                "barn spider",
                "garden spider",
                "black widow",
                "tarantula"
            ],
            "variations": [
                "on web","eating","inside","outside"
            ]
        },
        {
            "class": "snake",
            "prompt": "A photo of a {}, real-life setting.",
            "subclasses": [
                "diamondback rattlesnake",
                "thunder snake",
                "ringneck snake",
                "hognose snake",
                "green snake"
            ],
            "variations": [
                "lying down","curling up","eating","inside","outside","in water","on land","on rock"
            ]
        },
        {
            "class": "crab",
            "prompt": "A photo of a {}, real-life setting.",
            "subclasses": [
                "dungeness crab",
                "rock crab",
                "fiddler crab",
                "king crab",
                "hermit crab"
            ],
            "variations": [
                "on sea floor","on rock","in tank","in bowl","being held by human","on sand","submerged"
            ]
        },
        {
            "class": "beetle",
            "prompt": "A photo of a {}, real-life setting.",
            "subclasses": [
                "tiger beetle",
                "ground beetle",
                "long-horned beetle",
                "dung beetle",
                "weevil"
            ],
            "variations": [
                "on rock","on leaf","on ground","on tree","on branch","on grass","on sand"
            ]
        }
    ]
    
    
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
        
        # create folder for class in ../data/fake if it doesn't exist
        Path(f"../data/fake/{image_class}").mkdir(parents=True, exist_ok=True)
        

        for subclass in class_obj["subclasses"]:
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
    # generate_all_images()
    
    test_function()
    
