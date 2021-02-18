import torch
import os
from PIL import Image
from PIL.Image import Image as ImageType
from torchvision import transforms
import shutil
from typing import List
from datetime import timedelta

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

INPUT_DIR = '/home/aidar/Projects/PycharmProjects/ml_lab_test/data/input'
OUTPUT_DIR = '/home/aidar/Projects/PycharmProjects/ml_lab_test/data/output'
TENSOR_PATH = '/home/aidar/Projects/PycharmProjects/ml_lab_test/data/tensor/tensor.pt'
MODEL_PATH = '/home/aidar/Projects/PycharmProjects/ml_lab_test/models/model.pt'
THRESHOLD = 0.1

default_args = {
    'owner': 'Aidar Garikhanov',
    'depends_on_past': False,
    'email': ['a1d4r@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
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


def transform(image: ImageType):
    if image.mode == 'RGB':
        return transform_RGB(image)
    elif image.mode == 'L':
        return transform_L(image)
    else:
        raise NotImplementedError(f'Mode {image.mode} is not supported')


@dag(default_args=default_args, schedule_interval=None, start_date=days_ago(2))
def filter_images():
    """Filter medical images"""
    @task()
    def load_and_preprocess() -> List[str]:
        """Preprocess images."""
        images = []
        image_names = []
        for filename in os.listdir(INPUT_DIR):
            if filename.endswith('.png'):
                image_names.append(filename)
                images.append(Image.open(os.path.join(INPUT_DIR, filename)))
        tensor = torch.cat([transform(image).unsqueeze(0) for image in images], dim=0)
        torch.save(tensor, TENSOR_PATH)
        return image_names

    @task()
    def filter(image_names: List[str]) -> List[str]:
        """Filter images."""
        # Run loaded script
        tensor = torch.load(TENSOR_PATH)
        loaded_script = torch.jit.load(MODEL_PATH)
        result = loaded_script(tensor)

        return [filename
                for tensor, filename in zip(result, image_names)
                if tensor[0].item() < THRESHOLD]

    @task()
    def save(filtered_images: List[str]) -> None:
        """Save images."""
        for file in filtered_images:
            shutil.copy2(os.path.join(INPUT_DIR, file), OUTPUT_DIR)

    image_names = load_and_preprocess()
    filtered_images = filter(image_names)
    save(filtered_images)


filter_images_dag = filter_images()
