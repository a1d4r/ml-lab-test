import torch
import os
from PIL import Image
from torchvision import transforms
import shutil


INPUT_DIR = './input'
OUTPUT_DIR = './output'
MODEL_PATH = 'models/model.pt'
THRESHOLD = 0.1

# Load script
loaded_script = torch.jit.load(MODEL_PATH)

# Load images
images = []
for file in os.listdir(INPUT_DIR):
    if file.endswith('.png'):
        images.append((file, Image.open(os.path.join(INPUT_DIR, file))))

# Define transforms
transform_RGB = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Resize((224, 224))
])

transform_L = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Resize((224, 224))
])


def transform(image):
    if image.mode == 'RGB':
        return transform_RGB(image)
    elif image.mode == 'L':
        return transform_L(image)
    else:
        raise NotImplementedError(f'Mode {image.mode} is not supported')


# Apply transforms
tensor = torch.cat([transform(image).unsqueeze(0) for _, image in images], dim=0)

# Run loaded script
result = loaded_script(tensor)

# Filter images
filtered_images = [(file, image)
                   for tensor, (file, image) in zip(result, images)
                   if tensor[0].item() < THRESHOLD]

# Save filtered images
for file, _ in filtered_images:
    shutil.copy2(os.path.join(INPUT_DIR, file), OUTPUT_DIR)
