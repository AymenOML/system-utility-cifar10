import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

def load_and_preprocess_data(npy_dir='data'):
    os.makedirs(npy_dir, exist_ok=True)

    x_train_path = os.path.join(npy_dir, 'x_train.npy')
    y_train_path = os.path.join(npy_dir, 'y_train.npy')
    x_test_path  = os.path.join(npy_dir, 'x_test.npy')
    y_test_path  = os.path.join(npy_dir, 'y_test.npy')

    if all(os.path.exists(p) for p in [x_train_path, y_train_path, x_test_path, y_test_path]):
        print("Loading data from .npy files...")
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        x_test  = np.load(x_test_path)
        y_test  = np.load(y_test_path)
    else:
        print("Downloading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        print("Saving data to .npy files...")
        np.save(x_train_path, x_train)
        np.save(y_train_path, y_train)
        np.save(x_test_path,  x_test)
        np.save(y_test_path,  y_test)

    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0

    # One-hot encode
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test, 10)

    print(f"Training set: {x_train.shape}, Labels: {y_train_cat.shape}")
    print(f"Test set:     {x_test.shape}, Labels: {y_test_cat.shape}")

    return (x_train, y_train_cat), (x_test, y_test_cat)
