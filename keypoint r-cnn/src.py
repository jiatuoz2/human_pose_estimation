import torch
import torchvision
from torchvision import transforms
from PIL import Image

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

image = Image.open("./archive/Images/CCTV_1.png")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_image = transform(image).unsqueeze(0)