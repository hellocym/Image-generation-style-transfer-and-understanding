from image_gen import ImageGen
from image_transfer import ImageTransfer
from image_understanding import ImageUnderstanding

import argparse
import base64
import requests
from PIL import Image
from io import BytesIO

import os
import torchvision

import numpy as np

from dotenv import load_dotenv
load_dotenv()
API_KEY_IMAGE_GEN = os.getenv("API_KEY_IMAGE_GEN")
API_KEY_IMAGE_UNDERSTANDING = os.getenv("API_KEY_IMAGE_UNDERSTANDING")

# gradio web app
import gradio as gr

def image_transfer(content_image_url, style_image_url):
    config = argparse.Namespace(max_size=400, total_step=501, log_step=10, sample_step=500, style_weight=100, lr=0.03)
    image_transfer = ImageTransfer(config)
    img_with_style = image_transfer.transfer(content_image_url, style_image_url)
    img_path = 'output_2.png'
    torchvision.utils.save_image(img_with_style, img_path, nrow=1)
    img_base = base64.b64encode(open(img_path, 'rb').read()).decode('utf-8')
    image_understanding = ImageUnderstanding(api_key=API_KEY_IMAGE_UNDERSTANDING)
    return image_understanding.understanding(img_base)

def image_gen(prompt):
    image_gen = Image
    return image_gen.gen(prompt)

def image_understanding(img_base64):
    image_understanding = ImageUnderstanding
    return image_understanding.understanding(img_base64)

content_image_url = gr.inputs.Textbox(lines=1, label="Content Image URL")
style_image_url = gradio.components.inputs.Textbox(lines=1, label="Style Image URL")
image_transfer = gr.components.outputs.Image(label="Image Transfer")

gr.Interface(fn=image_transfer, inputs=[content_image_url, style_image_url], outputs=image_transfer).launch()

if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--max_size', type=int, default=400)
    # parser.add_argument('--total_step', type=int, default=501)
    # parser.add_argument('--log_step', type=int, default=10)
    # parser.add_argument('--sample_step', type=int, default=500)
    # parser.add_argument('--style_weight', type=float, default=100)
    # parser.add_argument('--lr', type=float, default=0.03)
    # config = parser.parse_args()
    # print(config)
    # image_gen = ImageGen(api_key=API_KEY_IMAGE_GEN)
    # content_image_url = image_gen.gen("一只可爱的狐狸")
    # style_image_url = image_gen.gen("一幅梵高的画")
    
    # print(content_image_url)
    # print(style_image_url)

    # image_transfer = ImageTransfer(config)
    # img_with_style = image_transfer.transfer(content_image_url, style_image_url)

    # torchvision.utils.save_image(img_with_style, 'output_2.png', nrow=1)

    # img_base = base64.b64encode(open('output_2.png', 'rb').read()).decode('utf-8')
 

    # image_understanding = ImageUnderstanding(api_key=API_KEY_IMAGE_UNDERSTANDING)
    # print(image_understanding.understanding(img_base))
    