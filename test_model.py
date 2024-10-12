#doubt in this code and model:
import torch
import torchvision.transforms as transforms
from torchvision import models

# Ensure the code runs on CPU
device = torch.device("cpu")

# Load your pre-trained model
model = models.resnet50(pretrained=True)
model = model.to(device)  # Move the model to CPU

# Example transformation for your input data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Example function to process an input image
from PIL import Image

def process_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Example forward pass through the model
def predict(image_path):
    model.eval()
    image = process_image(image_path)
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(image)
    return output

# Example usage
if __name__ == "__main__":
    image_path = 'path_to_image.jpg'  # Replace with your image path
    result = predict(image_path)
    print(result)
# hav eto rewrite the above code