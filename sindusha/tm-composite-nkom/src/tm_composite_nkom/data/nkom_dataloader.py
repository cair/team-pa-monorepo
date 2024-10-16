from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np
from loguru import logger
from skimage.filters import threshold_otsu
from tmu.data import TMUDataset
from tqdm import tqdm

class NKOMDatasetError(Exception):
    """Custom exception for NKOM dataset errors."""
    pass

class NKOMDataset(TMUDataset):
    def __init__(self,
                 data_dir: Path,
                 cache_dir: Path,
                 img_size: Tuple[int, int],
                 classes: Dict[str, int]):
        super().__init__()
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.img_size = img_size
        self.classes = classes
        self._check_data_availability()

    def _check_data_availability(self):
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
        return dataset #self._convert_to_binary(dataset).astype(np.uint32)

    def _load_or_process_data(self, split: str):
        cache_file_X = self.cache_dir / f'{split}_data_X.npy'
        cache_file_y = self.cache_dir / f'{split}_data_y.npy'

        if cache_file_X.exists() and cache_file_y.exists():
            logger.info(f"Loading cached data from {cache_file_X} and {cache_file_y}")
            with cache_file_X.open('rb') as fx, cache_file_y.open('rb') as fy:
                X = np.load(fx)
                y = np.load(fy)
        else:
            logger.info(f"Processing data from {self.data_dir / split}")
            X, y = self._load_and_preprocess_data(self.data_dir / split)

            logger.info(f"Saving processed data to {cache_file_X} and {cache_file_y}")
            with cache_file_X.open('wb') as fx, cache_file_y.open('wb') as fy:
                np.save(fx, X)
                np.save(fy, y)

        return X, y

    def _load_and_preprocess_data(self, data_dir: Path):
        X, y = [], []
        for class_name, label in self.classes.items():
            class_dir = data_dir / class_name
            for img_path in tqdm(list(class_dir.glob('*')), desc=f"Loading {class_name}"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.img_size)
                X.append(img)

                y.append(label)

        # Reshape X to (samples, dim1, dim2, depth)
        X = np.array(X).reshape(-1, self.img_size[0], self.img_size[1], 1)
        y = np.array(y)

        return X, y

    def _convert_to_binary(self, X_images):
        binary_images = []
        for image in X_images:
            threshold = threshold_otsu(image.squeeze())  # Remove the channel dimension for Otsu
            binary = (image > threshold).astype(np.uint32)
            binary_images.append(binary)
        return np.array(binary_images)