import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tmu.composite.callbacks.base import TMCompositeCallback
from tmu.composite.composite import TMComposite
from tmu.composite.components.adaptive_thresholding import AdaptiveThresholdingComponent
from tmu.composite.components.color_thermometer_scoring import ColorThermometerComponent
from tmu.composite.components.histogram_of_gradients import HistogramOfGradientsComponent
from tmu.composite.config import TMClassifierConfig
from tmu.models.classification.vanilla_classifier import TMClassifier
from data.nkom_dataloader import NKOMDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent.absolute()

class Settings(BaseSettings):
    experiment_type: str = "nkom"
    platform: str = "CPU"
    epochs: int = 30
    num_clauses: int = 100
    t_parameter: int = 500
    image_width: int = 100
    image_height: int = 100
    data_dir: str = "data-split"
    cache_dir: str = "cache"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix=""
    )

def plot_accuracy(train_acc, val_acc, test_acc, output_path):
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

def run_nkom_experiment(config: Settings, data):
    # Train and evaluate model
    num_clauses, T, epochs = config.num_clauses, config.t_parameter, config.epochs
    logger.info(f"Training model with {num_clauses} clauses, T={T}, and {epochs} epochs")

    tm = TMClassifier(
        number_of_clauses=num_clauses,
        T=T,
        s=5.0,
        patch_dim=(20, 20),
        max_included_literals=32,
        platform=config.platform,
        weighted_clauses=True
    )

    train_acc_history, val_acc_history, test_acc_history = [], [], []

    for epoch in range(epochs):
        tm.fit(data['x_train'], data['y_train'], epochs=1, incremental=True)

        train_acc = 100 * accuracy_score(data['y_train'], tm.predict(data['x_train']))
        val_acc = 100 * accuracy_score(data['y_val'], tm.predict(data['x_val']))
        test_acc = 100 * accuracy_score(data['y_test'], tm.predict(data['x_test']))

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        test_acc_history.append(test_acc)

        # Final evaluation
        y_pred = tm.predict(data['x_test'])
        cm = confusion_matrix(data['y_test'], y_pred)
        accuracy = accuracy_score(data['y_test'], y_pred)
        precision = precision_score(data['y_test'], y_pred)
        recall = recall_score(data['y_test'], y_pred)
        f1 = f1_score(data['y_test'], y_pred)

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

    # Plot accuracy
    plot_accuracy(train_acc_history, val_acc_history, test_acc_history, CURRENT_DIR / 'nkom_accuracy_plot.png')

def run_composite_experiment(config: Settings, data):
    checkpoint_path = Path("checkpoints")
    checkpoint_path.mkdir(exist_ok=True)

    composite_path = checkpoint_path / "composite"
    composite_path.mkdir(exist_ok=True)

    component_path = checkpoint_path / "components"
    component_path.mkdir(exist_ok=True)

    data_train = dict(X=data['x_train'], Y=data['y_train'])
    data_test = dict(X=data['x_test'], Y=data['y_test'])

    class TMCompositeCheckpointCallback(TMCompositeCallback):
        def on_epoch_component_end(self, component, epoch, logs=None):
            component.save(component_path / f"{component}-{epoch}.pkl")

    class TMCompositeEvaluationCallback(TMCompositeCallback):
        def __init__(self, data):
            super().__init__()
            self.best_acc = 0.0
            self.data = data

    # Define the composite model
    composite_model = TMComposite(
        components=[
            AdaptiveThresholdingComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=2000, T=500, s=10.0, max_included_literals=32,
                platform=config.platform, weighted_clauses=True, patch_dim=(10, 10),
            ), epochs=config.epochs),

            ColorThermometerComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=2000, T=1500, s=2.5, max_included_literals=32,
                platform=config.platform, weighted_clauses=True, patch_dim=(3, 3),
            ), resolution=8, epochs=config.epochs),

            ColorThermometerComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=2000, T=1500, s=2.5, max_included_literals=32,
                platform=config.platform, weighted_clauses=True, patch_dim=(4, 4),
            ), resolution=8, epochs=config.epochs),

            HistogramOfGradientsComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=2000, T=50, s=10.0, max_included_literals=32,
                platform=config.platform, weighted_clauses=False
            ), epochs=config.epochs)
        ],
        use_multiprocessing=False
    )

    # Train the composite model
    composite_model.fit(
        data=data_train,
        callbacks=[
            TMCompositeCheckpointCallback(),
            TMCompositeEvaluationCallback(data=data_test)
        ]
    )

    preds = composite_model.predict(data=data_test)

    y_true = data_test["Y"].flatten()
    for k, v in preds.items():
        comp_acc = 100 * (v == y_true).mean()
        logger.info(f"{k} Accuracy: {comp_acc:.1f}%")

def main():
    # Load environment variables
    load_dotenv()

    # Load settings
    config = Settings()

    # Constants
    image_dimensions = (config.image_width, config.image_height)
    classes = {'jam': 0, 'unintent_jam': 1}
    data_dir = CURRENT_DIR / config.data_dir
    cache_dir = CURRENT_DIR / config.cache_dir

    # Load data using the custom NKOMDataset
    data = NKOMDataset(
        data_dir=data_dir,
        cache_dir=cache_dir,
        img_size=image_dimensions,
        classes=classes
    ).get()

    if config.experiment_type == "nkom":
        run_nkom_experiment(config, data)
    elif config.experiment_type == "composite":
        run_composite_experiment(config, data)
    else:
        print(f"Unknown experiment type: {config.experiment_type}")

if __name__ == "__main__":
    main()