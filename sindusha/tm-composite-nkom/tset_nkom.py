import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from pathlib import Path
from tmu.models.classification.vanilla_classifier import TMClassifier
from loguru import logger

# Constants
IMG_SIZE = (100, 100)
CLASSES = {'jam': 0, 'unintent_jam': 1}
CURRENT_DIR = Path(__file__).parent.absolute()
DATA_DIR = CURRENT_DIR / 'data-split'
CACHE_DIR = CURRENT_DIR / 'cache'

# Configure logger
logger.add(CURRENT_DIR / "log/file_{time}.log", rotation="500 MB")

def load_and_preprocess_data(data_dir):
    X, y = [], []
    for class_name, label in CLASSES.items():
        class_dir = data_dir / class_name
        for img_path in tqdm(list(class_dir.glob('*')), desc=f"Loading {class_name}"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(label)
    return np.array(X).astype(np.uint32) , np.array(y).astype(np.uint32)

def convert_to_binary(X_images):
    binary_images = []
    for image in X_images:
        threshold = threshold_otsu(image)
        binary = (image > threshold).astype(np.uint32)
        binary_images.append(binary)
    return np.array(binary_images)

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, num_clauses, T, epochs):
    s = 5.0
    patch_dim = (20, 20)
    max_included_literals = 32
    platform = "CUDA"
    weighted_clauses = True
    tm = TMClassifier(
        num_clauses,
        T,
        s,
        patch_dim=patch_dim,
        max_included_literals=max_included_literals,
        platform=platform,
        weighted_clauses=weighted_clauses
    )
    
    train_acc_history, val_acc_history, test_acc_history = [], [], []
    
    for epoch in range(epochs):
        tm.fit(X_train, y_train, epochs=1, incremental=True)
        
        train_acc = 100 * accuracy_score(y_train, tm.predict(X_train))
        val_acc = 100 * accuracy_score(y_val, tm.predict(X_val))
        test_acc = 100 * accuracy_score(y_test, tm.predict(X_test))
        
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        test_acc_history.append(test_acc)
        
        # Final evaluation
        y_pred = tm.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        log_message = (
            f"Parameters: clauses={num_clauses}, T={T}, patch_size=20x20\n"
            f"Epoch {epoch+1}:\n"
            f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%\n"
            f"Confusion Matrix:\n{cm}\n"
            f"Accuracy: {accuracy*100:.2f}%\n"
            f"Precision: {precision*100:.2f}%\n"
            f"Recall: {recall*100:.2f}%\n"
            f"F1 Score: {f1*100:.2f}%"
        )
        logger.info(log_message)

    return tm, train_acc_history, val_acc_history, test_acc_history

def plot_accuracy(train_acc, val_acc, test_acc):
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy over Epochs')
    plt.legend()
    plt.savefig(CURRENT_DIR / 'accuracy_plot.png')
    logger.info(f"Accuracy plot saved to {CURRENT_DIR / 'accuracy_plot.png'}")

def load_or_process_data(data_dir, cache_file_X, cache_file_y):
    if cache_file_X.exists() and cache_file_y.exists():
        logger.info(f"Loading cached data from {cache_file_X} and {cache_file_y}")
        with cache_file_X.open('rb') as fx, cache_file_y.open('rb') as fy:
            X = np.load(fx)
            y = np.load(fy)
        return X, y
    
    logger.info(f"Processing data from {data_dir}")
    X, y = load_and_preprocess_data(data_dir)
    
    logger.info(f"Saving processed data to {cache_file_X} and {cache_file_y}")
    with cache_file_X.open('wb') as fx, cache_file_y.open('wb') as fy:
        np.save(fx, X)
        np.save(fy, y)
    
    return X, y

def main():
    logger.info("Starting main function")

    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(exist_ok=True)
    logger.info(f"Cache directory: {CACHE_DIR}")

    # Load and preprocess data
    X_train, y_train = load_or_process_data(DATA_DIR / 'train', CACHE_DIR / 'train_data_X.npy', CACHE_DIR / 'train_data_y.npy')
    X_val, y_val = load_or_process_data(DATA_DIR / 'val', CACHE_DIR / 'val_data_X.npy', CACHE_DIR / 'val_data_y.npy')
    X_test, y_test = load_or_process_data(DATA_DIR / 'test', CACHE_DIR / 'test_data_X.npy', CACHE_DIR / 'test_data_y.npy')
    
    # Convert to binary
    logger.info("Converting images to binary")
    X_train = convert_to_binary(X_train)
    X_val = convert_to_binary(X_val)
    X_test = convert_to_binary(X_test)
    
    # Train and evaluate model
    num_clauses, T, epochs = 100, 500, 1000
    logger.info(f"Training model with {num_clauses} clauses, T={T}, and {epochs} epochs")
    tm, train_acc, val_acc, test_acc = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, num_clauses, T, epochs)
    
    # Plot accuracy
    plot_accuracy(train_acc, val_acc, test_acc)
    

    logger.info("Main function completed")

if __name__ == "__main__":
    main()