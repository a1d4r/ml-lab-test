import torch
import os
from PIL import Image
from torchvision import transforms


INPUT_DIR = './input'
OUTPUT_DIR = './output'
MODEL_PATH = 'model.pt'

# Load script
loaded_script = torch.jit.load(MODEL_PATH)

# Load images
images = []
for file in os.listdir(INPUT_DIR):
    if file.endswith('.png'):

        images.append(Image.open(os.path.join(INPUT_DIR, file)))

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
tensor = torch.cat([transform(image).unsqueeze(0) for image in images], dim=0)

# Run loaded script
result = loaded_script(tensor)
print(result)
print(result[0][0].item())
