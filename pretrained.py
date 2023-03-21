# In this Python file, the trained model is loaded and 
# the image is preprocessed so that it can be used in the model

import torch
import torch.nn
from torchvision import models, transforms
from PIL import Image

# Setup and run the CUDA operations. If CUDA is not available, use the CPU.
device =  "cuda" if torch.cuda.is_available() else "cpu"
labels = './label/imagenet_classes.txt'
imagenet_labels = dict(enumerate(open(labels)))
resnet = models.resnet101(pretrained=True)
# Set where to run the model and matrix calculation
model = resnet.to(device)
model.eval()

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])
])

def preprocess(image):
    # Webcam frames are numpy array format, therefore transform back to PIL image
    image = Image.fromarray(image)
    image = data_transforms(image)
    image = torch.unsqueeze(image,0)
    return image    