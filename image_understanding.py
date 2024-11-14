#视频理解示例、上传视频URL
from zhipuai import ZhipuAI

import base64
import os
from zhipuai import ZhipuAI

from dotenv import load_dotenv
load_dotenv()
API_KEY_IMAGE_GEN = os.getenv("API_KEY_IMAGE_GEN")
API_KEY_IMAGE_UNDERSTANDING = os.getenv("API_KEY_IMAGE_UNDERSTANDING")

class ImageUnderstanding:
    def __init__(self, api_key):
        self.client = ZhipuAI(api_key=api_key)

    def understanding(self, img_base64):
        # with open(image, 'rb') as img_file:
        #     img_base = base64.b64encode(img_file.read()).decode('utf-8')
        response = self.client.chat.completions.create(
            model="glm-4v-plus",  # 填写需要调用的模型名称
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": "请描述这个图片，从风格和内容两方面描述。在风格上，这幅画可能是按照哪位艺术家的风格生成的？"
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message


if __name__ == "__main__":
    img_path = 'output_2.png'
    img_base = base64.b64encode(open(img_path, 'rb').read()).decode('utf-8')
    image_understanding = ImageUnderstanding(api_key=API_KEY_IMAGE_UNDERSTANDING)
    print(image_understanding.understanding(img_base))