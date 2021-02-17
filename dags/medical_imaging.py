import json
import torch
import os
from PIL import Image
from torchvision import transforms
import shutil
from typing import List, Tuple
from datetime import timedelta

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

INPUT_DIR = './input'
OUTPUT_DIR = './output'
MODEL_PATH = 'model.pt'
THRESHOLD = 0.1

default_args = {
    'owner': 'Aidar Garikhanov',
    'depends_on_past': False,
    'email': ['a1d4r@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

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


@dag(default_args=default_args, schedule_interval=None, start_date=days_ago(2))
def filter_images():
    """Filter medical images"""
    @task(multiple_outputs=True)
    def load() -> List[Tuple[str, Image]]:
        """Load images"""
        images = []
        for file in os.listdir(INPUT_DIR):
            if file.endswith('.png'):
                images.append((file, Image.open(os.path.join(INPUT_DIR, file))))
        return images

    @task(multiple_outputs=True)
    def transform(images: List[Tuple[str, Image]]) -> torch.Tensor:
        """Transform images."""
        return torch.cat([transform(image).unsqueeze(0) for _, image in images], dim=0)

    @task(multiple_outputs=True)
    def filter(tensor, images: List[Image]) -> List[Tuple[str, Image]]:
        """Filter images."""
        # Run loaded script
        loaded_script = torch.jit.load(MODEL_PATH)
        result = loaded_script(tensor)

        return [(file, image)
                for tensor, (file, image) in zip(result, images)
                if tensor[0].item() < THRESHOLD]

    @task()
    def save(filtered_images: List[Tuple[str, Image]]) -> None:
        """Save images."""
        for file, _ in filtered_images:
            shutil.copy2(os.path.join(INPUT_DIR, file), OUTPUT_DIR)

    images = load()
    tensor = transform(images)
    filtered_images = filter(tensor, images)
    save(filtered_images)


filter_images_dag = filter_images()
