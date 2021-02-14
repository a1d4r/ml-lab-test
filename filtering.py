import torch
import numpy as np
import os
from PIL import Image

INPUT_DIR = './input'
OUTPUT_DIR = './output'
MODEL_PATH = 'model.pt'

loaded_script = torch.jit.load(MODEL_PATH)

images = []
for file in os.listdir(INPUT_DIR):
    if file.endswith('.png'):
        images.append(Image.open(os.path.join(INPUT_DIR, file)))

print(images)
