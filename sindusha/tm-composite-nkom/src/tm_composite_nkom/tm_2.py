import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy.ndimage import rotate
from skimage.filters import threshold_otsu
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

CURRENT_DIR = Path(__file__).parent.absolute()

from tmu.composite.callbacks.base import TMCompositeCallback
from tmu.composite.components.base import TMComponent
from tmu.composite.composite import TMComposite
from tmu.composite.components.adaptive_thresholding import AdaptiveThresholdingComponent
from tmu.composite.components.color_thermometer_scoring import ColorThermometerComponent
from tmu.composite.components.histogram_of_gradients import HistogramOfGradientsComponent
from tmu.composite.config import TMClassifierConfig
from tmu.models.classification.vanilla_classifier import TMClassifier
from data.nkom_dataloader import NKOMDataset, NKOMDatasetError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor(ABC):
    @abstractmethod
    def process(self, images):
        pass

    def __str__(self):
        return self.__class__.__name__

class RotationPreprocessor(ImagePreprocessor):
    def __init__(self, rotation_angle):
        self.rotation_angle = rotation_angle

    def process(self, images):
        if self.rotation_angle == 0:
            return images
        rotated_images = []
        for image in images:
            rotated = rotate(image, self.rotation_angle, reshape=False, order=1, mode='constant', cval=0)
            rotated_images.append(rotated)
        return np.array(rotated_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.rotation_angle}"

class OtsuThresholdingPreprocessor(ImagePreprocessor):
    def process(self, images):
        binary_images = []
        for image in images:
            threshold = threshold_otsu(image)
            binary = (image > threshold).astype(np.uint32)
            binary_images.append(binary)
        return np.array(binary_images)

class CompositePreprocessor(ImagePreprocessor):
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def process(self, images):
        for preprocessor in self.preprocessors:
            images = preprocessor.process(images)
        return images

    def __str__(self):
        return f"{self.__class__.__name__}_{'_'.join(str(p) for p in self.preprocessors)}"

class FlexibleComponent(TMComponent):
    def __init__(self, model_cls, model_config, preprocessor, **kwargs):
        super().__init__(model_cls=model_cls, model_config=model_config, **kwargs)
        self.preprocessor = preprocessor

    def preprocess(self, data: dict):
        super().preprocess(data=data)
        X = data["X"]
        Y = data["Y"]
        X_processed = self.preprocessor.process(X)

        # ensure that the processed images are in the correct shape
        # batch, height, width, channels
        if len(X_processed.shape) == 3:
            X_processed = X_processed[..., np.newaxis]

        return dict(
            X=X_processed,
            Y=Y,
        )

    def __str__(self):
        return f"{self.__class__.__name__}_{self.preprocessor}"

class AdaptiveThresholdingPreprocessor(ImagePreprocessor):
    def __init__(self, block_size=11, C=2):
        self.block_size = block_size
        self.C = C

    def process(self, images):
        processed_images = []
        for image in images:
            processed = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, self.block_size, self.C)
            processed_images.append(processed)
        return np.array(processed_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.block_size}_{self.C}"

class CannyEdgePreprocessor(ImagePreprocessor):
    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def process(self, images):
        processed_images = []
        for image in images:
            edges = cv2.Canny(image, self.low_threshold, self.high_threshold)
            processed_images.append(edges)
        return np.array(processed_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.low_threshold}_{self.high_threshold}"

class HistogramEqualizationPreprocessor(ImagePreprocessor):
    def process(self, images):
        processed_images = []
        for image in images:
            equalized = cv2.equalizeHist(image)
            processed_images.append(equalized)
        return np.array(processed_images)

    def __str__(self):
        return f"{self.__class__.__name__}"

class MorphologicalPreprocessor(ImagePreprocessor):
    def __init__(self, operation='open', kernel_size=5):
        self.operation = operation
        self.kernel_size = kernel_size
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def process(self, images):
        processed_images = []
        for image in images:
            if self.operation == 'open':
                processed = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)
            elif self.operation == 'close':
                processed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.kernel)
            processed_images.append(processed)
        return np.array(processed_images)

    def __str__(self):
        return f"{self.__class__.__name__}_{self.operation}_{self.kernel_size}"

class ExperimentType(str):
    nkom = "nkom"
    composite = "composite"



class Settings(BaseSettings):
    experiment_type: str = ExperimentType.composite
    platform: str = "CPU"
    epochs: int = 100
    num_clauses: int = 100
    t_parameter: int = 500
    image_width: int = 100
    image_height: int = 100
    data_dir: str = "data-split"
    cache_dir: str = "cache"
    train_percentage: Optional[float] = None  # New parameter

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix=""
    )

def plot_accuracy(train_acc: List[float], val_acc: List[float], test_acc: List[float], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy over Epochs')
    plt.legend()
    plt.savefig(output_path)
    logger.info(f"Accuracy plot saved to {output_path}")

def run_nkom_experiment(config: Settings, data: Dict[str, np.ndarray[np.uint32]]) -> None:

    logger.info(f"Training model with {config.num_clauses} clauses, T={config.t_parameter}, and {config.epochs} epochs")

    # Summary of data distribution:
    # count unique values in the y_train array
    unique, counts = np.unique(data['y_train'], return_counts=True)
    logger.info(f"Training data distribution: {dict(zip(unique, counts))}")

    unique, counts = np.unique(data['y_val'], return_counts=True)
    logger.info(f"Validation data distribution: {dict(zip(unique, counts))}")

    unique, counts = np.unique(data['y_test'], return_counts=True)
    logger.info(f"Test data distribution: {dict(zip(unique, counts))}")

    tm = TMClassifier(
        number_of_clauses=config.num_clauses,
        T=config.t_parameter,
        s=5.0,
        patch_dim=(20, 20),
        max_included_literals=32,
        platform=config.platform,
        weighted_clauses=True
    )

    train_acc_history: List[float] = []
    val_acc_history: List[float] = []
    test_acc_history: List[float] = []

    for epoch in range(config.epochs):
        tm.fit(
            data['x_train'],
            data['y_train'],
        )

        y_pred = tm.predict(data['x_test'])
        y_pred_val = tm.predict(data['x_val'])
        y_pred_train = tm.predict(data['x_train'])

        train_acc = 100 * accuracy_score(data['y_train'], y_pred_train)
        val_acc = 100 * accuracy_score(data['y_val'], y_pred_val)
        test_acc = 100 * accuracy_score(data['y_test'], y_pred)

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        test_acc_history.append(test_acc)


        cm = confusion_matrix(data['y_test'], y_pred)
        accuracy = accuracy_score(data['y_test'], y_pred)
        precision = precision_score(data['y_test'], y_pred, average='weighted')
        recall = recall_score(data['y_test'], y_pred, average='weighted')
        f1 = f1_score(data['y_test'], y_pred, average='weighted')

        log_message = (
            f"Epoch {epoch+1}:\n"
            f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%\n"
            f"Confusion Matrix:\n{cm}\n"
            f"Accuracy: {accuracy*100:.2f}%\n"
            f"Precision: {precision*100:.2f}%\n"
            f"Recall: {recall*100:.2f}%\n"
            f"F1 Score: {f1*100:.2f}%"
        )
        logger.info(log_message)

    plot_accuracy(train_acc_history, val_acc_history, test_acc_history, CURRENT_DIR / 'nkom_accuracy_plot.png')

def run_composite_experiment(config: Settings, dataset: Dict[str, np.ndarray]) -> None:
    checkpoint_path = Path("checkpoints")
    checkpoint_path.mkdir(exist_ok=True)

    composite_path = checkpoint_path / "composite"
    composite_path.mkdir(exist_ok=True)

    component_path = checkpoint_path / "components"
    component_path.mkdir(exist_ok=True)

    data_train = dict(X=dataset['x_train'], Y=dataset['y_train'])
    data_test = dict(X=dataset['x_test'], Y=dataset['y_test'])

    class TMCompositeCheckpointCallback(TMCompositeCallback):
        def __init__(self, component_path: Path, data_test: Dict[str, Any]):
            super().__init__()
            self.component_path = component_path
            self.progress_bars = {}
            self.component_id_names = {}
            self.component_accuracies = {}
            self.data_test = data_test

        def on_train_composite_begin(self, composite, logs=None, **kwargs):
            for i, component in enumerate(composite.components):
                component_str = str(component)
                self.progress_bars[component.uuid] = tqdm(
                    total=component.epochs,
                    desc=self._get_description(component, 0),
                    position=i,
                    leave=True
                )
                self.component_id_names[component.uuid] = f"Component {i}: {component_str}"
                self.component_accuracies[component.uuid] = 0.0

            # Add an overall progress bar
            total_epochs = sum(component.epochs for component in composite.components)
            self.progress_bars['overall'] = tqdm(
                total=total_epochs,
                desc="Overall Progress",
                position=len(composite.components),
                leave=True
            )

        def on_epoch_component_begin(self, component, epoch, logs=None, **kwargs):
            pbar = self.progress_bars.get(component.uuid)
            if pbar:
                pbar.set_description(self._get_description(component, epoch + 1))

        def on_epoch_component_end(self, component: TMComponent, epoch: int, logs: Dict = None, **kwargs) -> None:
            component.save(self.component_path / f"{component}-{epoch}.pkl")

            pbar = self.progress_bars.get(component.uuid)
            if pbar:
                # Update accuracy
                # model = component.model_instance
                # y_pred = model.predict(self.data_test['X'])
                # y_true = self.data_test['Y'].flatten()
                # acc = 100 * (y_pred == y_true).mean()
                # self.component_accuracies[component.uuid] = acc

                # Update progress bar
                pbar.update(1)
                pbar.set_description(self._get_description(component, epoch + 1))
                pbar.refresh()

            # Update overall progress
            overall_pbar = self.progress_bars.get('overall')
            if overall_pbar:
                overall_pbar.update(1)
                overall_pbar.refresh()

        def on_train_composite_end(self, composite, logs=None, **kwargs):
            print("\nTraining completed. Final component accuracies:")
            for component in composite.components:
                accuracy = self.component_accuracies.get(component.uuid, 0.0)
                print(f"{self.component_id_names[component.uuid]}: {accuracy:.1f}%")

            # Uncomment the following lines if you want to close the progress bars
            # for pbar in self.progress_bars.values():
            #     pbar.close()
            # print("\n" * (len(self.progress_bars) + 1))  # Move cursor below all progress bars

        def _get_description(self, component: TMComponent, current_epoch: int) -> str:
            if component.uuid not in self.component_accuracies:
                self.component_accuracies[component.uuid] = 0.0
            accuracy = self.component_accuracies[component.uuid]
            component_str = str(component)
            return f"{component_str} - Epoch {current_epoch}/{component.epochs} - Acc: {accuracy:.1f}%"


    composite_model = TMComposite(
        components=[
            # Multiple rotations with OtsuThresholding
            *[
                FlexibleComponent(
                    model_cls=TMClassifier,
                    model_config=TMClassifierConfig(
                        number_of_clauses=100,
                        T=500,
                        s=5.0,
                        max_included_literals=32,
                        platform=config.platform,
                        weighted_clauses=True,
                        patch_dim=(20, 20),
                    ),
                    preprocessor=CompositePreprocessor([
                        RotationPreprocessor(rotation_angle=angle),
                        OtsuThresholdingPreprocessor()
                    ]),
                    epochs=config.epochs
                )
                for angle in range(0, 315, 45)
            ],
            # Adaptive Thresholding with different block sizes
            *[
                FlexibleComponent(
                    model_cls=TMClassifier,
                    model_config=TMClassifierConfig(
                        number_of_clauses=100,
                        T=500,
                        s=5.0,
                        max_included_literals=32,
                        platform=config.platform,
                        weighted_clauses=True,
                        patch_dim=(20, 20),
                    ),
                    preprocessor=AdaptiveThresholdingPreprocessor(block_size=block_size),
                    epochs=config.epochs
                )
                for block_size in [3, 7, 11, 15]
            ],
            # Edge Detection
            FlexibleComponent(
                model_cls=TMClassifier,
                model_config=TMClassifierConfig(
                    number_of_clauses=100,
                    T=500,
                    s=5.0,
                    max_included_literals=32,
                    platform=config.platform,
                    weighted_clauses=True,
                    patch_dim=(20, 20),
                ),
                preprocessor=CannyEdgePreprocessor(low_threshold=50, high_threshold=150),
                epochs=config.epochs
            ),
            # Histogram Equalization followed by Otsu Thresholding
            FlexibleComponent(
                model_cls=TMClassifier,
                model_config=TMClassifierConfig(
                    number_of_clauses=100,
                    T=500,
                    s=5.0,
                    max_included_literals=32,
                    platform=config.platform,
                    weighted_clauses=True,
                    patch_dim=(20, 20),
                ),
                preprocessor=CompositePreprocessor([
                    HistogramEqualizationPreprocessor(),
                    OtsuThresholdingPreprocessor()
                ]),
                epochs=config.epochs
            ),

            # Morphological Operations
            FlexibleComponent(
                model_cls=TMClassifier,
                model_config=TMClassifierConfig(
                    number_of_clauses=100,
                    T=500,
                    s=5.0,
                    max_included_literals=32,
                    platform=config.platform,
                    weighted_clauses=True,
                    patch_dim=(20, 20),
                ),
                preprocessor=MorphologicalPreprocessor(operation='open', kernel_size=5),
                epochs=config.epochs
            )
        ],
        use_multiprocessing=True
    )

    composite_model.fit(
        data=data_train,
        callbacks=[
            TMCompositeCheckpointCallback(data_test=data_test, component_path=component_path)
        ]
    )

    preds = composite_model.predict(data=data_test)

    y_true = data_test["Y"].flatten()
    for k, v in preds.items():
        comp_acc = 100 * (v == y_true).mean()
        logger.info(f"{k} Accuracy: {comp_acc:.1f}%")

def main() -> None:
    load_dotenv()

    config = Settings()

    image_dimensions: Tuple[int, int] = (config.image_width, config.image_height)
    classes: Dict[str, int] = {'jam': 0, 'unintent_jam': 1}
    data_dir = CURRENT_DIR / config.data_dir
    cache_dir = CURRENT_DIR / config.cache_dir

    try:
        if config.experiment_type == "nkom":
            data = NKOMDataset(
                data_dir=data_dir,
                cache_dir=cache_dir,
                img_size=image_dimensions,
                classes=classes,
                train_percentage=1.0,
                config_convert_to_binary=True
            ).get()

            run_nkom_experiment(config, data)
        elif config.experiment_type == "composite":
            data = NKOMDataset(
                data_dir=data_dir,
                cache_dir=cache_dir,
                img_size=image_dimensions,
                classes=classes,
                train_percentage=1.0,
                config_convert_to_binary=False
            ).get()

            run_composite_experiment(config, data)
        else:
            logger.error(f"Unknown experiment type: {config.experiment_type}")
    except NKOMDatasetError as e:
        logger.error(f"Error loading NKOM dataset: {e}")
        print("Please follow the instructions above to download and set up the NKOM dataset.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()