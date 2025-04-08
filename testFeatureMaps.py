import torch
from models.yolo import Model
from models.common import VOneBlock
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load the model configuration
model_conv = Model('/home/wcortez/yolov5/models/yolov5l.yaml')

# Load and preprocess the input image
image_path = '/home/wcortez/datasets/coco/images/val2017/000000001000.jpg'
image = Image.open(image_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),  # Adjust based on input size requirement
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
x = preprocess(image).unsqueeze(0)  # Add batch dimension

# Initialize VOneBlock and run input through it
first_layer_vone = VOneBlock()
y_vone = first_layer_vone(x)

# Visualize feature maps
for i in range(64):  # Loop over the first 12 channels
    plt.figure(figsize=(12, 12))  # Set the size for each individual plot
    plt.imshow(y_vone[0, i].detach().cpu(), cmap='viridis')
    plt.axis('off')
    plt.savefig(f"maps/feature_map_channel_{i + 1}.png", bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to save memory

