# pre_download_dataset.py
import os
import numpy as np
from tensorflow.keras.datasets import cifar10

def pre_download(npy_dir='data'):
    """
    Downloads the CIFAR-10 dataset and saves it to .npy files.
    This should be run once on a login node with internet access.
    """
    os.makedirs(npy_dir, exist_ok=True)
    print("Downloading CIFAR-10 dataset...")
    try:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print("Download complete.")
        
        x_train_path = os.path.join(npy_dir, 'x_train.npy')
        y_train_path = os.path.join(npy_dir, 'y_train.npy')
        x_test_path  = os.path.join(npy_dir, 'x_test.npy')
        y_test_path  = os.path.join(npy_dir, 'y_test.npy')

        print("Saving data to .npy files...")
        np.save(x_train_path, x_train)
        np.save(y_train_path, y_train)
        np.save(x_test_path,  x_test)
        np.save(y_test_path,  y_test)
        print(f"Data successfully saved to the '{npy_dir}' directory.")
    except Exception as e:
        print(f"An error occurred during download or saving: {e}")
        print("Please check your internet connection and directory permissions.")

if __name__ == '__main__':
    pre_download()