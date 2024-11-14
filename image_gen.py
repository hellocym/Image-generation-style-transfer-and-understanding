from zhipuai import ZhipuAI
import requests

# load token from env
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY_IMAGE_GEN = os.getenv("API_KEY_IMAGE_GEN")


class ImageGen:
    def __init__(self, api_key):
        self.client = ZhipuAI(api_key=api_key)
  
    def gen(self, prompt):
        response = self.client.images.generations(
            model="cogview-3-plus", #填写需要调用的模型编码
            prompt=prompt,
        )
        url = response.data[0].url
        # download image from url as binary
        # response = requests.get(url)
        return url

if __name__ == "__main__":
    image_gen = ImageGen(api_key=API_KEY_IMAGE_GEN)
    print(image_gen.gen("一只可爱的狐狸"))
