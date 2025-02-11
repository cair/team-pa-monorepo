import logging
import math
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from filters.functions import FlexibleComponent, OtsuThresholdingPreprocessor, CompositePreprocessor, RotationPreprocessor, \
    HistogramEqualizationPreprocessor, GaussianBlurPreprocessor, SobelEdgePreprocessor, \
    LaplacianEdgePreprocessor, BilateralFilterPreprocessor, AdaptiveThresholdingPreprocessor, CannyEdgePreprocessor, \
    FrequencyBandThresholdingPreprocessor, PeakDetectionPreprocessor, SpectralContrastEnhancementPreprocessor, \
    TimeFrequencyEdgeDetectionPreprocessor, HarmonicStructureEnhancementPreprocessor, \
    AdaptiveThresholdingWithSkeletonizationPreprocessor

CURRENT_DIR = Path(__file__).parent.absolute()

from tm_composite_nkom.data.sindusha_dataloader import SindushaDataset
from tmu.composite.callbacks.base import TMCompositeCallback
from tmu.composite.components.base import TMComponent
from tmu.composite.composite import TMComposite
from tmu.composite.config import TMClassifierConfig
from tmu.models.classification.vanilla_classifier import TMClassifier
from data.nkom_dataloader import NKOMDataset, NKOMDatasetError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class ExperimentType(str):
    nkom = "nkom"
    composite = "composite"
    sindusha = "sindusha"


class Settings(BaseSettings):
    experiment_type: str = ExperimentType.sindusha
    platform: str = "CUDA"
    epochs: int = 100
    num_clauses: int = 100
    t_parameter: int = 500
    image_width: int = 100
    image_height: int = 100
    data_dir: str = "datasets/jamming_hard"
    cache_dir: str = "cache"
    train_percentage: Optional[float] = None  # New parameter

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix=""
    )




class TMCompositeCheckpointCallback(TMCompositeCallback):
    def __init__(self, component_path: Path, data_test: Dict[str, Any]):
        super().__init__()
        self.component_path = component_path
        self.progress_bars = {}
        self.component_id_names = {}
        self.component_accuracies = {}
        self.data_test = data_test
        self.reported_epochs = {}  # Track which epochs have been reported

    def on_train_composite_begin(self, composite, logs=None, **kwargs):
        for i, component in enumerate(composite.components):
            component_str = str(component)
            self.reported_epochs[component.uuid] = set()  # Initialize empty set for this component
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


    def on_epoch_component_end(self, component: TMComponent, epoch: int, logs: Dict = dict(), **kwargs) -> None:
        # Skip if we've already reported this epoch for this component
        if epoch in self.reported_epochs.get(component.uuid, set()):
            return

        component.save(self.component_path / f"{component}-{epoch}.pkl")

        pbar = self.progress_bars.get(component.uuid)
        if pbar:
            # Update accuracy every 15 epochs
            if epoch == 100:
                model = component.model_instance
                y_pred = model.predict(self.data_test['X'])
                y_true = self.data_test['Y'].flatten()
                acc = 100 * (y_pred == y_true).mean()
                self.component_accuracies[component.uuid] = acc

            # Update progress bar
            pbar.update(1)
            pbar.set_description(self._get_description(component, epoch + 1))
            pbar.refresh()

        # Update overall progress
        overall_pbar = self.progress_bars.get('overall')
        if overall_pbar:
            overall_pbar.update(1)
            overall_pbar.refresh()

        # Mark this epoch as reported
        if component.uuid not in self.reported_epochs:
            self.reported_epochs[component.uuid] = set()
        self.reported_epochs[component.uuid].add(epoch)

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

def run_nkom_experiment(config: Settings, data: Dict[str, np.ndarray[Any, np.dtype[np.uint32]]]) -> None:

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

        train_acc_history.append(float(train_acc))
        val_acc_history.append(float(val_acc))
        test_acc_history.append(float(test_acc))


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

def plot_misclassifications(X_misclassified, y_true_misclassified, y_pred_misclassified, images_per_figure=25):
    num_misclassified = len(X_misclassified)
    num_figures = math.ceil(num_misclassified / images_per_figure)

    for fig_num in range(num_figures):
        plt.figure(figsize=(20, 20))
        for i in range(images_per_figure):
            idx = fig_num * images_per_figure + i
            if idx >= num_misclassified:
                break

            plt.subplot(5, 5, i + 1)
            plt.imshow(X_misclassified[idx].squeeze(), cmap='gray')
            plt.title(f"True: {y_true_misclassified[idx]}, Pred: {y_pred_misclassified[idx]}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"plot_{fig_num + 1}.png")
        plt.savefig(f"plot_{fig_num + 1}.pdf")
        plt.close()


def run_composite_experiment(config: Settings, dataset: Dict[str, np.ndarray]) -> None:
    checkpoint_path = Path("checkpoints")
    checkpoint_path.mkdir(exist_ok=True)

    composite_path = checkpoint_path / "composite"
    composite_path.mkdir(exist_ok=True)

    component_path = checkpoint_path / "components"
    component_path.mkdir(exist_ok=True)

    data_train = dict(X=dataset['x_train'], Y=dataset['y_train'])
    data_test = dict(X=dataset['x_test'], Y=dataset['y_test'])



    # composite_model = TMComposite(
    #     components=[
    #         # # Basic Thresholding Techniques
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=OtsuThresholdingPreprocessor(),
    #             epochs=config.epochs
    #         ),
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=AdaptiveThresholdingPreprocessor(block_size=11, C=2),
    #             epochs=config.epochs
    #         ),
    #
    #         # Rotations with Thresholding
    #         *[
    #             FlexibleComponent(
    #                 model_cls=TMClassifier,
    #                 model_config=TMClassifierConfig(
    #                     number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                     platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #                 ),
    #                 preprocessor=CompositePreprocessor([
    #                     RotationPreprocessor(rotation_angle=angle),
    #                     OtsuThresholdingPreprocessor()
    #                 ]),
    #                 epochs=config.epochs
    #             )
    #             for angle in [45, 90]
    #         ],
    #
    #         # Edge Detection Techniques
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=CannyEdgePreprocessor(low_threshold=50, high_threshold=150),
    #             epochs=config.epochs
    #         ),
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=SobelEdgePreprocessor(ksize=3),
    #             epochs=config.epochs
    #         ),
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=LaplacianEdgePreprocessor(ksize=3),
    #             epochs=config.epochs
    #         ),
    #
    #         # Frequency-based Techniques
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=FrequencyBandThresholdingPreprocessor(num_bands=10),
    #             epochs=config.epochs
    #         ),
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=PeakDetectionPreprocessor(min_distance=10),
    #             epochs=config.epochs
    #         ),
    #
    #         # Contrast Enhancement Techniques
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=CompositePreprocessor([
    #                 HistogramEqualizationPreprocessor(),
    #                 OtsuThresholdingPreprocessor()
    #             ]),
    #             epochs=config.epochs
    #         ),
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=SpectralContrastEnhancementPreprocessor(percentile=95),
    #             epochs=config.epochs
    #         ),
    #
    #         # Smoothing and Edge Detection Combinations
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=CompositePreprocessor([
    #                 GaussianBlurPreprocessor(kernel_size=(5, 5), sigma=0),
    #                 CannyEdgePreprocessor(low_threshold=50, high_threshold=150)
    #             ]),
    #             epochs=config.epochs
    #         ),
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=CompositePreprocessor([
    #                 BilateralFilterPreprocessor(d=9, sigmaColor=75, sigmaSpace=75),
    #                 SobelEdgePreprocessor(ksize=3)
    #             ]),
    #             epochs=config.epochs
    #         ),
    #
    #         # Time-Frequency Analysis
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=TimeFrequencyEdgeDetectionPreprocessor(sigma=1.0),
    #             epochs=config.epochs
    #         ),
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=HarmonicStructureEnhancementPreprocessor(min_angle=80, max_angle=100),
    #             epochs=config.epochs
    #         ),
    #
    #         # Advanced Combinations
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=CompositePreprocessor([
    #                 HistogramEqualizationPreprocessor(),
    #                 FrequencyBandThresholdingPreprocessor(num_bands=5),
    #                 TimeFrequencyEdgeDetectionPreprocessor(sigma=1.0)
    #             ]),
    #             epochs=config.epochs
    #         ),
    #         FlexibleComponent(
    #             model_cls=TMClassifier,
    #             model_config=TMClassifierConfig(
    #                 number_of_clauses=100, T=500, s=5.0, max_included_literals=32,
    #                 platform=config.platform, weighted_clauses=True, patch_dim=(20, 20),
    #             ),
    #             preprocessor=CompositePreprocessor([
    #                 GaussianBlurPreprocessor(kernel_size=(3, 3), sigma=0),
    #                 SpectralContrastEnhancementPreprocessor(percentile=90),
    #                 AdaptiveThresholdingWithSkeletonizationPreprocessor(block_size=35, offset=10)
    #             ]),
    #             epochs=config.epochs
    #         ),
    #     ],
    #     use_multiprocessing=True
    # )
    #
    # composite_model.fit(
    #     data=data_train,
    #     callbacks=[
    #         TMCompositeCheckpointCallback(data_test=data_test, component_path=component_path)
    #     ]
    # )

    # Define the parameter ranges
    clauses = [100] #, 200, 300, 500]
    T_values = [50] #, 200, 500]
    s_values = [10.0, 5.0] # 1.0, 5.0]
    patch_dims = [(20, 20)]  # (10, 10), (40, 40)
    # Create all combinations of parameters
    param_combinations = list(product(clauses, T_values, s_values, patch_dims))

    # Create a list to store all components
    components = []

    for num_clauses, T, s, patch_dim in param_combinations:
        component = FlexibleComponent(
            model_cls=TMClassifier,
            model_config=TMClassifierConfig(
                number_of_clauses=num_clauses,
                T=T,
                s=s,
                max_included_literals=32,
                platform=config.platform,
                weighted_clauses=True,
                patch_dim=patch_dim,
            ),
            preprocessor=OtsuThresholdingPreprocessor(),
            epochs=config.epochs
        )
        components.append(component)

    # Create the composite model
    composite_model = TMComposite(
        components=components,
        use_multiprocessing=True,
    )

    composite_model.fit(
        data=data_train,
        callbacks=[
            TMCompositeCheckpointCallback(
                data_test=data_test,
                component_path=component_path
            )
        ]
    )

    preds = composite_model.predict(data=data_test)
    y_true = data_test["Y"].flatten()

    for k, v in preds.items():
        comp_acc = 100 * (v == y_true).mean()
        logger.info(f"{k} Accuracy: {comp_acc:.1f}%")

        is_misclassification = (v != y_true)
        X_misclassified = data_test["X"][is_misclassification]
        y_true_misclassified = y_true[is_misclassification]
        y_pred_misclassified = v[is_misclassification]

        plot_misclassifications(X_misclassified, y_true_misclassified, y_pred_misclassified)



def main() -> None:
    load_dotenv()

    config = Settings()

    image_dimensions: Tuple[int, int] = (config.image_width, config.image_height)
    data_dir = CURRENT_DIR / config.data_dir
    cache_dir = CURRENT_DIR / config.cache_dir

    try:
        if config.experiment_type == "nkom":
            data = NKOMDataset(
                data_dir=data_dir,
                cache_dir=cache_dir,
                img_size=image_dimensions,
                classes={'jam': 0, 'unintent_jam': 1},
                train_percentage=0.8,
                config_convert_to_binary=True
            ).get()

            run_nkom_experiment(config, data)
        elif config.experiment_type == "composite":
            data = NKOMDataset(
                data_dir=data_dir,
                cache_dir=cache_dir,
                img_size=image_dimensions,
                classes={'jam': 0, 'unintent_jam': 1},
                train_percentage=0.8,
                config_convert_to_binary=False
            ).get()

            run_composite_experiment(config, data)
        elif config.experiment_type == "sindusha":
            dataset = SindushaDataset(
                data_dir=data_dir,
                cache_dir=cache_dir,
                img_size=(100, 100),
                classes={
                    'DME': 0,
                    'NB': 1,
                    'NoJam': 2,
                    'SingleAM': 3,
                    'SingleChirp': 4,
                    'SingleFM': 5
                },
                train_percentage=0.8,
                config_convert_to_binary=True,
                val_split=0.1
            )
            data = dataset.get()
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
