import logging
import math
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from filters.functions import (
    FlexibleComponent,
    OtsuThresholdingPreprocessor,
    CompositePreprocessor,
    RotationPreprocessor,
    HistogramEqualizationPreprocessor,
    GaussianBlurPreprocessor,
    SobelEdgePreprocessor,
    LaplacianEdgePreprocessor,
    BilateralFilterPreprocessor,
    AdaptiveThresholdingPreprocessor,
    CannyEdgePreprocessor,
    FrequencyBandThresholdingPreprocessor,
    PeakDetectionPreprocessor,
    SpectralContrastEnhancementPreprocessor,
    TimeFrequencyEdgeDetectionPreprocessor,
    HarmonicStructureEnhancementPreprocessor,
    AdaptiveThresholdingWithSkeletonizationPreprocessor,
)
import progressbar

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
    platform: str = "CPU"
    epochs: int = 100
    num_clauses: int = 100
    t_parameter: int = 500
    image_width: int = 100
    image_height: int = 100
    data_dir: str = "datasets/sindusha_dataset"
    cache_dir: str = "cache"
    train_percentage: Optional[float] = None  # New parameter

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix=""
    )


class TMCompositeCheckpointCallback(TMCompositeCallback):
    def __init__(self, component_path: Path, data_test: Dict[str, Any]):
        super().__init__()
        self.component_path = component_path
        self.progress_bars = {}
        self.component_id_names = {}
        self.component_accuracies = {}
        self.data_test = data_test
        self.reported_epochs = {}
        self.component_counter = 0

        # Create progress log file
        self.log_file = Path("training_progress.txt")
        # Clear the file at start
        with open(self.log_file, "w") as f:
            f.write("Training Progress Log\n==================\n\n")

    def _log_progress(self, message: str):
        """Write progress to log file"""
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")

    def _get_short_name(self, component_str: str) -> str:
        """Create a shorter, readable name for the component."""
        # Handle composite preprocessors
        if component_str.startswith('CompositePreprocessor_'):
            parts = component_str.replace('CompositePreprocessor_', '').split('_')
            return f"Comp({'+'.join(p[:3] for p in parts[:2])})"

        # Handle other preprocessors
        parts = component_str.split('_')
        base_name = parts[0].replace('Preprocessor', '')

        # Special handling for Otsu with different parameters
        if base_name == 'OtsuThresholding':
            # Get the TMClassifier parameters from the component's model_config
            if hasattr(self, 'component_configs') and self.component_counter in self.component_configs:
                config = self.component_configs[self.component_counter]
                return f"Otsu(c{config.number_of_clauses},T{config.T},s{config.s})"
            # Fallback if no config found
            return f"Otsu#{self.component_counter}"

        # Handle other preprocessors
        if len(parts) > 1:
            params = '_'.join(parts[1:])
            return f"{base_name[:6]}({params})"
        return base_name[:10]

    def on_train_composite_begin(self, composite, logs=None, **kwargs):
        self.component_counter = 0

        # Store component configurations
        self.component_configs = {}
        for i, component in enumerate(composite.components, 1):
            if hasattr(component, 'model_config'):
                self.component_configs[i] = component.model_config

        header = "\nStarting training with the following components:"
        print(header)
        self._log_progress(header)

        # Log component information
        for i, component in enumerate(composite.components):
            self.component_counter = i + 1
            component_str = str(component)
            short_name = self._get_short_name(component_str)
            log_line = f"{i+1}. {short_name} -> {component_str}"
            print(log_line)
            self._log_progress(log_line)

        progress_header = "\nProgress bars:"
        print(progress_header)
        self._log_progress(progress_header)
        print("\n" * (len(composite.components) + 2))  # Make space for all progress bars

        # Reset counter for progress bars
        self.component_counter = 0

        # Create widgets for each component
        for i, component in enumerate(composite.components):
            self.component_counter = i + 1
            component_str = str(component)
            short_name = self._get_short_name(component_str)
            self.reported_epochs[component.uuid] = set()
            self.component_id_names[component.uuid] = f"Component {i}: {component_str}"
            self.component_accuracies[component.uuid] = 0.0

            widgets = [
                f'{i+1}. {short_name:<30}',  # Fixed width for alignment
                progressbar.Percentage(),
                ' |',
                progressbar.Bar(marker='█', left='', right=''),
                '| ',
                progressbar.Counter(format='%(value)d/%(max_value)d'),
                ' ',
                progressbar.ETA(),
                ' Acc: ',
                progressbar.DynamicMessage('accuracy')
            ]

            self.progress_bars[component.uuid] = progressbar.ProgressBar(
                max_value=component.epochs,
                widgets=widgets,
                term_width=100,  # Fixed width to prevent wrapping
                fd=open(os.devnull, 'w')  # Prevent default stdout
            )
            self.progress_bars[component.uuid].start()

        # Create overall progress bar
        total_epochs = sum(component.epochs for component in composite.components)
        overall_widgets = [
            'Overall Progress:'.ljust(30),
            progressbar.Percentage(),
            ' |',
            progressbar.Bar(marker='█', left='', right=''),
            '| ',
            progressbar.Counter(format='%(value)d/%(max_value)d'),
            ' ',
            progressbar.ETA()
        ]
        self.progress_bars["overall"] = progressbar.ProgressBar(
            max_value=total_epochs,
            widgets=overall_widgets,
            term_width=100,
            fd=open(os.devnull, 'w')  # Prevent default stdout
        )
        self.progress_bars["overall"].start()

        # Store the cursor position for updating bars
        self.start_line = len(composite.components) + 2

        # After creating all progress bars, do an initial display
        self._update_progress_display()

    def _update_progress_display(self):
        """Update all progress bars on screen"""
        # Move cursor up to the start of progress bars
        print(f"\033[{self.start_line}A")

        # Print each progress bar
        for component_uuid, pbar in self.progress_bars.items():
            if component_uuid != "overall":
                # Get the formatted progress bar string
                progress_str = pbar.value or 0
                percentage = (progress_str / pbar.max_value) * 100 if pbar.max_value else 0
                accuracy = self.component_accuracies.get(component_uuid, 0.0)

                # Create the progress bar display
                bar_width = 40
                filled = int(bar_width * percentage / 100)
                bar = '█' * filled + '-' * (bar_width - filled)

                # Format the line with proper padding
                name = pbar.widgets[0].strip()  # Get the component name
                line = (
                    f"{name} {percentage:3.0f}% |{bar}| "
                    f"{progress_str:4d}/{pbar.max_value:<4d} "  # Fixed width for numbers
                    f"ETA: {pbar.eta() if hasattr(pbar, 'eta') and pbar.eta() else '--:--:--'} "
                    f"Acc: {accuracy:5.1f}%"
                )
                print(f"{line:<120}")  # Fixed total width
                print("\033[K")  # Clear to end of line

        # Print overall progress
        overall_pbar = self.progress_bars["overall"]
        overall_progress = overall_pbar.value or 0
        overall_percentage = (overall_progress / overall_pbar.max_value) * 100 if overall_pbar.max_value else 0
        overall_bar_width = 40
        overall_filled = int(overall_bar_width * overall_percentage / 100)
        overall_bar = '█' * overall_filled + '-' * (overall_bar_width - overall_filled)

        overall_line = (
            f"{'Overall Progress:'.ljust(30)} "
            f"{overall_percentage:3.0f}% |{overall_bar}| "
            f"{overall_progress:4d}/{overall_pbar.max_value:<4d} "
            f"ETA: {overall_pbar.eta() if hasattr(overall_pbar, 'eta') and overall_pbar.eta() else '--:--:--'}"
        )
        print(f"{overall_line:<120}")

        # Move cursor back down
        print(f"\033[{self.start_line}B")

    def on_epoch_component_begin(self, component, epoch, logs=None, **kwargs):
        pbar = self.progress_bars.get(component.uuid)
        if pbar:
            pbar.update(
                epoch,  # Changed from epoch + 1 to epoch
                accuracy=self.component_accuracies.get(component.uuid, 0.0)
            )
            self._update_progress_display()  # Update display immediately

    def on_epoch_component_end(
        self, component: TMComponent, epoch: int, metrics: Dict = None, **kwargs
    ) -> None:
        if epoch in self.reported_epochs.get(component.uuid, set()):
            return

        pbar = self.progress_bars.get(component.uuid)
        if pbar:
            if metrics and "accuracy" in metrics:
                self.component_accuracies[component.uuid] = metrics["accuracy"]
                # Log progress to file
                component_str = self._get_short_name(str(component))
                log_message = (
                    f"Component {component_str} - "
                    f"Epoch {epoch + 1}/{component.epochs} - "
                    f"Accuracy: {metrics['accuracy']:.2f}%"
                )
                self._log_progress(log_message)

            pbar.update(
                epoch + 1,
                accuracy=self.component_accuracies.get(component.uuid, 0.0)
            )

            if (epoch + 1) % 10 == 0:
                component.save(self.component_path / f"{component}-{epoch+1}.pkl")
                self._log_progress(f"Checkpoint saved for {component_str} at epoch {epoch + 1}")

            # Update overall progress
            overall_pbar = self.progress_bars.get("overall")
            if overall_pbar:
                current_value = overall_pbar.value or 0
                overall_pbar.update(current_value + 1)

            # Update display
            self._update_progress_display()

        if component.uuid not in self.reported_epochs:
            self.reported_epochs[component.uuid] = set()
        self.reported_epochs[component.uuid].add(epoch)

    def on_train_composite_end(self, composite, logs=None, **kwargs):
        # Finish all progress bars
        for pbar in self.progress_bars.values():
            pbar.finish()

        final_results = "\nTraining completed. Final component accuracies:"
        print(final_results)
        self._log_progress(final_results)

        for component in composite.components:
            accuracy = self.component_accuracies.get(component.uuid, 0.0)
            result_line = f"{self.component_id_names[component.uuid]}: {accuracy:.1f}%"
            print(result_line)
            self._log_progress(result_line)


def plot_accuracy(
    train_acc: List[float],
    val_acc: List[float],
    test_acc: List[float],
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.plot(test_acc, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy over Epochs")
    plt.legend()
    plt.savefig(output_path)
    logger.info(f"Accuracy plot saved to {output_path}")


def run_nkom_experiment(
    config: Settings, data: Dict[str, np.ndarray[Any, np.dtype[np.uint32]]]
) -> None:

    logger.info(
        f"Training model with {config.num_clauses} clauses, T={config.t_parameter}, and {config.epochs} epochs"
    )

    # Summary of data distribution:
    # count unique values in the y_train array
    unique, counts = np.unique(data["y_train"], return_counts=True)
    logger.info(f"Training data distribution: {dict(zip(unique, counts))}")

    unique, counts = np.unique(data["y_val"], return_counts=True)
    logger.info(f"Validation data distribution: {dict(zip(unique, counts))}")

    unique, counts = np.unique(data["y_test"], return_counts=True)
    logger.info(f"Test data distribution: {dict(zip(unique, counts))}")

    tm = TMClassifier(
        number_of_clauses=config.num_clauses,
        T=config.t_parameter,
        s=5.0,
        patch_dim=(20, 20),
        max_included_literals=32,
        platform=config.platform,
        weighted_clauses=True,
    )

    train_acc_history: List[float] = []
    val_acc_history: List[float] = []
    test_acc_history: List[float] = []

    for epoch in range(config.epochs):
        tm.fit(
            data["x_train"],
            data["y_train"],
        )

        y_pred = tm.predict(data["x_test"])
        y_pred_val = tm.predict(data["x_val"])
        y_pred_train = tm.predict(data["x_train"])

        train_acc = 100 * accuracy_score(data["y_train"], y_pred_train)
        val_acc = 100 * accuracy_score(data["y_val"], y_pred_val)
        test_acc = 100 * accuracy_score(data["y_test"], y_pred)

        train_acc_history.append(float(train_acc))
        val_acc_history.append(float(val_acc))
        test_acc_history.append(float(test_acc))

        cm = confusion_matrix(data["y_test"], y_pred)
        accuracy = accuracy_score(data["y_test"], y_pred)
        precision = precision_score(data["y_test"], y_pred, average="weighted")
        recall = recall_score(data["y_test"], y_pred, average="weighted")
        f1 = f1_score(data["y_test"], y_pred, average="weighted")

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

    plot_accuracy(
        train_acc_history,
        val_acc_history,
        test_acc_history,
        CURRENT_DIR / "nkom_accuracy_plot.png",
    )


def plot_misclassifications(
    X_misclassified, y_true_misclassified, y_pred_misclassified, images_per_figure=25
):
    num_misclassified = len(X_misclassified)
    num_figures = math.ceil(num_misclassified / images_per_figure)

    for fig_num in range(num_figures):
        plt.figure(figsize=(20, 20))
        for i in range(images_per_figure):
            idx = fig_num * images_per_figure + i
            if idx >= num_misclassified:
                break

            plt.subplot(5, 5, i + 1)
            plt.imshow(X_misclassified[idx].squeeze(), cmap="gray")
            plt.title(
                f"True: {y_true_misclassified[idx]}, Pred: {y_pred_misclassified[idx]}"
            )
            plt.axis("off")

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

    data_train = dict(X=dataset["x_train"], Y=dataset["y_train"])
    data_test = dict(X=dataset["x_test"], Y=dataset["y_test"])

    # limit to 1000 samples
    # data_train = dict(X=dataset["x_train"][:1], Y=dataset["y_train"][:1])
    # data_test = dict(X=dataset["x_test"][:1], Y=dataset["y_test"][:1])

    components = [
        # # Basic Thresholding Techniques
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
            preprocessor=OtsuThresholdingPreprocessor(),
            epochs=config.epochs,
        ),
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
            preprocessor=AdaptiveThresholdingPreprocessor(block_size=11, C=2),
            epochs=config.epochs,
        ),
        # Rotations with Thresholding
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
                preprocessor=CompositePreprocessor(
                    [
                        RotationPreprocessor(rotation_angle=angle),
                        OtsuThresholdingPreprocessor(),
                    ]
                ),
                epochs=config.epochs,
            )
            for angle in [45, 90]
        ],
        # Edge Detection Techniques
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
            epochs=config.epochs,
        ),
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
            preprocessor=SobelEdgePreprocessor(ksize=3),
            epochs=config.epochs,
        ),
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
            preprocessor=LaplacianEdgePreprocessor(ksize=3),
            epochs=config.epochs,
        ),
        # Frequency-based Techniques
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
            preprocessor=FrequencyBandThresholdingPreprocessor(num_bands=10),
            epochs=config.epochs,
        ),
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
            preprocessor=PeakDetectionPreprocessor(min_distance=10),
            epochs=config.epochs,
        ),
        # Contrast Enhancement Techniques
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
            preprocessor=CompositePreprocessor(
                [
                    HistogramEqualizationPreprocessor(),
                    OtsuThresholdingPreprocessor(),
                ]
            ),
            epochs=config.epochs,
        ),
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
            preprocessor=SpectralContrastEnhancementPreprocessor(percentile=95),
            epochs=config.epochs,
        ),
        # Smoothing and Edge Detection Combinations
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
            preprocessor=CompositePreprocessor(
                [
                    GaussianBlurPreprocessor(kernel_size=(5, 5), sigma=0),
                    CannyEdgePreprocessor(low_threshold=50, high_threshold=150),
                ]
            ),
            epochs=config.epochs,
        ),
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
            preprocessor=CompositePreprocessor(
                [
                    BilateralFilterPreprocessor(d=9, sigmaColor=75, sigmaSpace=75),
                    SobelEdgePreprocessor(ksize=3),
                ]
            ),
            epochs=config.epochs,
        ),
        # Time-Frequency Analysis
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
            preprocessor=TimeFrequencyEdgeDetectionPreprocessor(sigma=1.0),
            epochs=config.epochs,
        ),
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
            preprocessor=HarmonicStructureEnhancementPreprocessor(
                min_angle=80, max_angle=100
            ),
            epochs=config.epochs,
        ),
        # Advanced Combinations
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
            preprocessor=CompositePreprocessor(
                [
                    HistogramEqualizationPreprocessor(),
                    FrequencyBandThresholdingPreprocessor(num_bands=5),
                    TimeFrequencyEdgeDetectionPreprocessor(sigma=1.0),
                ]
            ),
            epochs=config.epochs,
        ),
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
            preprocessor=CompositePreprocessor(
                [
                    GaussianBlurPreprocessor(kernel_size=(3, 3), sigma=0),
                    SpectralContrastEnhancementPreprocessor(percentile=90),
                    AdaptiveThresholdingWithSkeletonizationPreprocessor(
                        block_size=35, offset=10
                    ),
                ]
            ),
            epochs=config.epochs,
        ),
    ]

    # # Define the parameter ranges
    clauses = [100, 200, 300, 500]
    T_values = [50, 200, 500]
    s_values = [10.0, 5.0, 1.0, 5.0]
    patch_dims = [(20, 20)]  # (10, 10), (40, 40)
    # Create all combinations of parameters
    param_combinations = list(product(clauses, T_values, s_values, patch_dims))

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
            epochs=config.epochs,
        )
        components.append(component)

    # Create the composite model
    composite_model = TMComposite(
        components=components,
        use_multiprocessing=True,
        test_frequency=10,
        max_workers=os.cpu_count(),
    )

    composite_model.fit(
        data=data_train,
        test_data=data_test,  # Pass test data to fit
        callbacks=[
            TMCompositeCheckpointCallback(
                data_test=data_test,  # This is now optional since workers do testing
                component_path=component_path,
            )
        ],
    )

    # composite_model.fit(
    #     data=data_train,
    #     callbacks=[
    #         TMCompositeCheckpointCallback(
    #             data_test=data_test, component_path=component_path
    #         )
    #     ],
    # )

    preds = composite_model.predict(data=data_test)
    y_true = data_test["Y"].flatten()

    for k, v in preds.items():
        comp_acc = 100 * (v == y_true).mean()
        logger.info(f"{k} Accuracy: {comp_acc:.1f}%")

        is_misclassification = v != y_true
        X_misclassified = data_test["X"][is_misclassification]
        y_true_misclassified = y_true[is_misclassification]
        y_pred_misclassified = v[is_misclassification]

        plot_misclassifications(
            X_misclassified, y_true_misclassified, y_pred_misclassified
        )


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
                classes={"jam": 0, "unintent_jam": 1},
                train_percentage=0.8,
                config_convert_to_binary=True,
            ).get()

            run_nkom_experiment(config, data)
        elif config.experiment_type == "composite":
            data = NKOMDataset(
                data_dir=data_dir,
                cache_dir=cache_dir,
                img_size=image_dimensions,
                classes={"jam": 0, "unintent_jam": 1},
                train_percentage=0.8,
                config_convert_to_binary=False,
            ).get()

            run_composite_experiment(config, data)
        elif config.experiment_type == "sindusha":
            dataset = SindushaDataset(
                data_dir=data_dir,
                cache_dir=cache_dir,
                img_size=(100, 100),
                classes={
                    "DME": 0,
                    "NB": 1,
                    "NoJam": 2,
                    "SingleAM": 3,
                    "SingleChirp": 4,
                    "SingleFM": 5,
                },
                train_percentage=0.8,
                config_convert_to_binary=False,
                val_split=0.1,
            )
            data = dataset.get()
            run_composite_experiment(config, data)
        else:
            logger.error(f"Unknown experiment type: {config.experiment_type}")
    except NKOMDatasetError as e:
        logger.error(f"Error loading NKOM dataset: {e}")
        print(
            "Please follow the instructions above to download and set up the NKOM dataset."
        )
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
