import os
from bs4 import BeautifulSoup as bs
import requests


def get_image_urls(html):
    soup = bs(html, 'html.parser')
    image_urls = []
    for link in soup.find_all('a'):
        
        if link.get('href') and 'https://cdn.discordapp.com/attachments/' in link.get('href') and '.png' in link.get('href'):
            image_urls.append(link.get('href'))
            
        
    # create folder called /inference_images/fake
    os.makedirs('inference_images/fake', exist_ok=True)
        
    for img in image_urls:
        # download image to folder called /inference_images/fake
        image_content = requests.get(img).content
        print(img)
        save_path = img.split("/")[-1].split('?')[0]
        with open(f'inference_images/fake/{save_path}', 'wb') as f:
            f.write(image_content)
        
    return image_urls

if __name__ == '__main__':
    html = open('discord_sdxl.html').read()
    get_image_urls(html)