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
    
    image = generate_image(pipe, "The Paradox of Achilles and the Tortoise is one of a number of theoretical discussions of movement put forward by the Greek philosopher Zeno of Elea in the 5th century BCE. It begins with the great hero Achilles challenging a tortoise to a footrace. To keep things fair, he agrees to give the tortoise a head start of, say, 500 meters. When the race begins, Achilles unsurprisingly starts running at a speed much faster than the tortoise, so that by the time he has reached the 500-meter mark, the tortoise has only walked 50 meters further than him. But by the time Achilles has reached the 550-meter mark, the tortoise has walked another 5 meters. And by the time he has reached the 555-meter mark, the tortoise has walked another 0.5 meters, then 0.25 meters, then 0.125 meters, and so on. This process continues again and again over an infinite series of smaller and smaller distances, with the tortoise always moving forwards while Achilles always plays catch up.  Logically, this seems to prove that Achilles can never overtake the tortoise—whenever he reaches somewhere the tortoise has been, he will always have some distance still left to go no matter how small it might be. Except, of course, we know intuitively that he can overtake the tortoise. The trick here is not to think of Zeno’s Achilles Paradox in terms of distances and races, but rather as an example of how any finite value can always be divided an infinite number of times, no matter how small its divisions might become.")
    image.save("test.jpeg")


if __name__ == "__main__":
    # generate_all_images()
    
    test_function()
    
