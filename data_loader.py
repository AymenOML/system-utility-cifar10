# In data_loader.py

import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

# Add mmap_mode and preprocess arguments
def load_and_preprocess_data(npy_dir='data', mmap_mode=None, preprocess=True):
    os.makedirs(npy_dir, exist_ok=True)

    x_train_path = os.path.join(npy_dir, 'x_train.npy')
    y_train_path = os.path.join(npy_dir, 'y_train.npy')
    x_test_path  = os.path.join(npy_dir, 'x_test.npy')
    y_test_path  = os.path.join(npy_dir, 'y_test.npy')

    if all(os.path.exists(p) for p in [x_train_path, y_train_path, x_test_path, y_test_path]):
        print("Loading data from .npy files...")
        # Use mmap_mode when loading the arrays
        x_train = np.load(x_train_path, mmap_mode=mmap_mode)
        y_train = np.load(y_train_path, mmap_mode=mmap_mode)
        x_test  = np.load(x_test_path, mmap_mode=mmap_mode)
        y_test  = np.load(y_test_path, mmap_mode=mmap_mode)
    else:
        # This part remains the same
        print("Downloading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print("Saving data to .npy files...")
        np.save(x_train_path, x_train)
        np.save(y_train_path, y_train)
        np.save(x_test_path,  x_test)
        np.save(y_test_path,  y_test)

    # Only preprocess if the flag is True
    if preprocess:
        x_train = x_train.astype('float32') / 255.0
        x_test  = x_test.astype('float32') / 255.0
        y_train = to_categorical(y_train, 10)
        y_test  = to_categorical(y_test, 10)

    print(f"Training set: {x_train.shape}, Labels: {y_train.shape}")
    print(f"Test set:     {x_test.shape}, Labels: {y_test.shape}")

    return (x_train, y_train), (x_test, y_test)