import torch
from models.common import VOneBlock
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load and preprocess the input image
image_path = '/home/wcortez/datasets/coco/images/val2017/000000001000.jpg'
image = Image.open(image_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
x = preprocess(image).unsqueeze(0)  # Add batch dimension
x.requires_grad_()  # Enable gradient tracking for input

# Initialize VOneBlock and pass the image through it
first_layer_vone = VOneBlock()
y_vone = first_layer_vone(x)

# Choose a specific feature map (e.g., sum of all feature maps) for the saliency map
output = y_vone.sum()  # Summing all feature map values to get a scalar

# Backpropagate to get the saliency map for the input image
output.backward()

# Get the absolute gradient values for the saliency map
saliency = x.grad.abs().squeeze().max(dim=0)[0].detach().cpu()

# Plot the saliency map
plt.figure(figsize=(10, 10))
plt.imshow(saliency, cmap='hot')
plt.axis('off')
plt.colorbar()
plt.savefig("saliencyMaps/fig1.png")