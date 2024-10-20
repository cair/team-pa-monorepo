import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
from loguru import logger
from skimage.filters import threshold_otsu
from tmu.data import TMUDataset
from tqdm import tqdm
import random

class NKOMDatasetError(Exception):
    """Custom exception for NKOM dataset errors."""

class NKOMDataset(TMUDataset):
    def __init__(
            self,
            data_dir: Path,
            cache_dir: Path,
            img_size: Tuple[int, int],
            classes: Dict[str, int],
            train_percentage: Optional[float] = None,
            config_convert_to_binary: bool = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.img_size = img_size
        self.classes = classes
        self.train_percentage = train_percentage
        self.cache_prefix = self.generate_cache_prefix()
        self.config_convert_to_binary = config_convert_to_binary
        self.check_data_availability()

    def generate_cache_prefix(self) -> str:
        """Generate a unique cache prefix based on the dataset parameters."""
        params = f"img_{self.img_size[0]}x{self.img_size[1]}_train_{self.train_percentage or 'full'}"
        return hashlib.md5(params.encode()).hexdigest()[:8]

    def check_data_availability(self):
        if not self.data_dir.exists():
            raise NKOMDatasetError(
                f"Data directory {self.data_dir} does not exist. "
                "Please download the NKOM dataset and place it in the correct directory. "
                "You can download the dataset from any authorized users. "
                f"After downloading, extract the contents to {self.data_dir}."
            )

        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                raise NKOMDatasetError(
                    f"The {split} directory is missing in {self.data_dir}. "
                    "Please ensure you have the correct directory structure: "
                    f"{self.data_dir}/train, {self.data_dir}/val, {self.data_dir}/test"
                )

            for class_name in self.classes.keys():
                class_dir = split_dir / class_name
                if not class_dir.exists() or not any(class_dir.iterdir()):
                    raise NKOMDatasetError(
                        f"The {class_name} class directory is missing or empty in {split_dir}. "
                        "Please ensure you have downloaded the complete dataset and "
                        "maintained the correct directory structure."
                    )

    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        train_data = self._load_or_process_data('train')
        val_data = self._load_or_process_data('val')
        test_data = self._load_or_process_data('test')

        return {
            'x_train': train_data[0],
            'y_train': train_data[1],
            'x_val': val_data[0],
            'y_val': val_data[1],
            'x_test': test_data[0],
            'y_test': test_data[1]
        }

    def _transform(self, name: str, dataset: np.ndarray) -> np.ndarray:
        if name.startswith('y'):
            return dataset.astype(np.uint32)

        if self.config_convert_to_binary:
            return self.convert_to_binary(dataset)

        return dataset

    def _load_or_process_data(self, split: str):
        cache_file_x = self.cache_dir / f'{self.cache_prefix}_{split}_data_x.npy'
        cache_file_y = self.cache_dir / f'{self.cache_prefix}_{split}_data_y.npy'

        if cache_file_x.exists() and cache_file_y.exists():
            logger.info(f"Loading cached data from {cache_file_x} and {cache_file_y}")
            with cache_file_x.open('rb') as fx, cache_file_y.open('rb') as fy:
                x = np.load(fx)
                y = np.load(fy)
        else:
            logger.info(f"Processing data from {self.data_dir / split}")
            x, y = self._load_and_preprocess_data(self.data_dir / split)

            logger.info(f"Saving processed data to {cache_file_x} and {cache_file_y}")
            with cache_file_x.open('wb') as fx, cache_file_y.open('wb') as fy:
                np.save(fx, x)
                np.save(fy, y)

        if split == 'train' and self.train_percentage is not None:
            num_samples = int(len(x) * self.train_percentage)
            indices = random.sample(range(len(x)), num_samples)
            x = x[indices]
            y = y[indices]
            logger.info(f"Using {self.train_percentage * 100}% of training data: {len(x)} samples")

        return x, y

    def _load_and_preprocess_data(self, data_dir: Path):
        x, y = [], []
        for class_name, label in self.classes.items():
            class_dir = data_dir / class_name
            for img_path in tqdm(list(class_dir.glob('*')), desc=f"Loading {class_name}"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.img_size)
                x.append(img)
                y.append(label)

        # Reshape x to (samples, dim1, dim2, depth)
        x = np.array(x).reshape(-1, self.img_size[0], self.img_size[1], 1)
        y = np.array(y)

        return x, y

    @staticmethod
    def convert_to_binary(x_images):
        binary_images = []
        for image in x_images:
            threshold = threshold_otsu(image.squeeze())  # Remove the channel dimension for Otsu
            binary = (image > threshold).astype(np.uint32)
            binary_images.append(binary)
        return np.array(binary_images)