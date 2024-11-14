from image_gen import ImageGen
from image_transfer import ImageTransfer
from image_understanding import ImageUnderstanding

import argparse
import base64
import requests
from PIL import Image
from io import BytesIO

import torchvision

import numpy as np

from dotenv import load_dotenv
load_dotenv()
API_KEY_IMAGE_GEN = os.getenv("API_KEY_IMAGE_GEN")
API_KEY_IMAGE_UNDERSTANDING = os.getenv("API_KEY_IMAGE_UNDERSTANDING")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_size', type=int, default=400)
    parser.add_argument('--total_step', type=int, default=501)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--style_weight', type=float, default=100)
    parser.add_argument('--lr', type=float, default=0.03)
    config = parser.parse_args()
    print(config)
    image_gen = ImageGen(api_key=API_KEY_IMAGE_GEN)
    content_image_url = image_gen.gen("一只可爱的狐狸")
    style_image_url = image_gen.gen("一幅梵高的画")
    
    print(content_image_url)
    print(style_image_url)

    image_transfer = ImageTransfer(config)
    img_with_style = image_transfer.transfer(content_image_url, style_image_url)

    torchvision.utils.save_image(img_with_style, 'output_2.png', nrow=1)

    img_base = base64.b64encode(open('output_2.png', 'rb').read()).decode('utf-8')
 

    image_understanding = ImageUnderstanding(api_key=API_KEY_IMAGE_UNDERSTANDING)
    print(image_understanding.understanding(img_base))
    