from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
from io import BytesIO
import requests
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY_IMAGE_GEN = os.getenv("API_KEY_IMAGE_GEN")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")

def load_image(image_path, transform=None, max_size=None, shape=None):
    """Load an image and convert it to a torch tensor."""
    image = Image.open(image_path)
    if len(image.mode) !=3:
        image=image.convert('RGB')
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)
    
    if shape:
        image = image.resize(shape, Image.LANCZOS)
    
    if transform:
        image = transform(image).unsqueeze(0)
    
    return image.to(device)


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28'] 
        self.vgg = models.vgg19(pretrained=True).features
        
    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features




class ImageTransfer:
    def __init__(self, config):
        self.config = config
        self.vgg = VGGNet().to(device).eval()

    def transfer(self, content_url, style_url):
        config = self.config
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                std=(0.229, 0.224, 0.225))])
        
 
        # convert to tensor
        content = requests.get(content_url).content
        style = requests.get(style_url).content
        content = Image.open(BytesIO(content))
        style = Image.open(BytesIO(style))

        content = transform(content).unsqueeze(0).to(device)
        style = transform(style).unsqueeze(0).to(device)


        # Load content and style images
        # content = load_image(config.content, transform, max_size=config.max_size)
        # style = load_image(config.style, transform, shape=[content.size(2), content.size(3)])
        
        # Initialize a target image with the content image
        target = content.clone().requires_grad_(True)
        
        optimizer = torch.optim.Adam([target], lr=config.lr, betas=[0.5, 0.999])
        vgg = self.vgg

        for parameter in vgg.parameters():
            parameter.requires_grad=False
        for step in range(config.total_step):
            
            # Extract multiple(5) conv feature vectors
            target_features = vgg(target)
            content_features = vgg(content)
            style_features = vgg(style)

            style_loss = 0
            content_loss = 0
            for f1, f2, f3 in zip(target_features, content_features, style_features):
                # Compute content loss with target and content images
                content_loss += torch.mean((f1 - f2)**2)

                # Reshape convolutional feature maps
                _, c, h, w = f1.size()
                f1 = f1.view(c, h * w)
                f3 = f3.view(c, h * w)

                # Compute gram matrix
                f1 = torch.mm(f1, f1.t())
                f3 = torch.mm(f3, f3.t())

                # Compute style loss with target and style images
                style_loss += torch.mean((f1 - f3)**2) / (c * h * w) 
            
            # Compute total loss, backprop and optimize
            loss = content_loss + config.style_weight * style_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % config.log_step == 0:
                print ('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}' 
                    .format(step+1, config.total_step, content_loss.item(), style_loss.item()))

            
        denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        img = target.clone().squeeze()
        img = denorm(img).clamp_(0, 1)
            
        return img




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--content', type=str, default='png/00026P.jpg')
    # parser.add_argument('--style', type=str, default='png/style3.png')
    parser.add_argument('--max_size', type=int, default=400)
    parser.add_argument('--total_step', type=int, default=501)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--style_weight', type=float, default=100)
    parser.add_argument('--lr', type=float, default=0.03)
    config = parser.parse_args()
    print(config)
    from image_gen import ImageGen
    image_gen = ImageGen(api_key=API_KEY_IMAGE_GEN)
    content_image_url = image_gen.gen("一只可爱的狐狸")
    style_image_url = image_gen.gen("一幅梵高的画")
    print(content_image_url)
    print(style_image_url)



    image_transfer = ImageTransfer(config)
    img_with_style = image_transfer.transfer(content_image_url, style_image_url)

    # Save the generated image
    torchvision.utils.save_image(img_with_style, 'output_2.png', nrow=1)