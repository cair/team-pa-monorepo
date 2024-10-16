import argparse
import logging
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from tmu.data import TMUDataset
from tmu.composite.callbacks.base import TMCompositeCallback
from tmu.composite.composite import TMComposite
from tmu.composite.components.adaptive_thresholding import AdaptiveThresholdingComponent
from tmu.composite.components.color_thermometer_scoring import ColorThermometerComponent
from tmu.composite.components.histogram_of_gradients import HistogramOfGradientsComponent
from tmu.composite.config import TMClassifierConfig
from tmu.models.classification.vanilla_classifier import TMClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = (100, 100)
CLASSES = {'jam': 0, 'unintent_jam': 1}
CURRENT_DIR = Path(__file__).parent.absolute()
DATA_DIR = CURRENT_DIR / 'data-split'
CACHE_DIR = CURRENT_DIR / 'cache'

class NKOMDataset(TMUDataset):
    def __init__(self):
        super().__init__()
        self.data_dir = DATA_DIR
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
        self.img_size = IMG_SIZE
        self.classes = CLASSES

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

def run_nkom_experiment(args, data):
    # Train and evaluate model
    num_clauses, T, epochs = args.num_clauses, args.T, args.epochs
    logger.info(f"Training model with {num_clauses} clauses, T={T}, and {epochs} epochs")
    
    tm = TMClassifier(
        number_of_clauses=num_clauses,
        T=T,
        s=5.0,
        patch_dim=(20, 20),
        max_included_literals=32,
        platform=args.platform,
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

def run_composite_experiment(args, data):
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
                platform=args.platform, weighted_clauses=True, patch_dim=(10, 10),
            ), epochs=args.epochs),

            ColorThermometerComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=2000, T=1500, s=2.5, max_included_literals=32,
                platform=args.platform, weighted_clauses=True, patch_dim=(3, 3),
            ), resolution=8, epochs=args.epochs),

            ColorThermometerComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=2000, T=1500, s=2.5, max_included_literals=32,
                platform=args.platform, weighted_clauses=True, patch_dim=(4, 4),
            ), resolution=8, epochs=args.epochs),

            HistogramOfGradientsComponent(TMClassifier, TMClassifierConfig(
                number_of_clauses=2000, T=50, s=10.0, max_included_literals=32,
                platform=args.platform, weighted_clauses=False
            ), epochs=args.epochs)
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
    parser = argparse.ArgumentParser(description="Run Tsetlin Machine experiments")
    parser.add_argument("--experiment", choices=["nkom", "composite"], required=True, help="Type of experiment to run")
    parser.add_argument("--platform", default="CPU", type=str, help="Platform to run on (CPU or CUDA)")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs")
    parser.add_argument("--num_clauses", default=100, type=int, help="Number of clauses (for NKOM experiment)")
    parser.add_argument("--T", default=500, type=int, help="T parameter (for NKOM experiment)")
    
    args = parser.parse_args()

    # Load data using the custom NKOMDataset
    data = NKOMDataset().get()
    
    if args.experiment == "nkom":
        run_nkom_experiment(args, data)
    elif args.experiment == "composite":
        run_composite_experiment(args, data)

if __name__ == "__main__":
    main()