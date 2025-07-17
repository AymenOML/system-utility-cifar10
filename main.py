from data_loader import load_and_preprocess_data
from model import build_cnn_model
from train import train_model

import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Loading and preprocessing CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    print("Dataset loaded and preprocessed successfully.\n")

    print("Building CNN model...")
    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=10)
    print("Model built successfully.\n")

    print("Model summary:")
    model.summary()
    print()

    print("Starting training...")
    history = train_model(model, x_train, y_train, x_test, y_test)
    print("Training completed.\n")

    print("Evaluating model on test set...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}\n")

    print("Plotting training history...")
    history_df = pd.DataFrame(history.history)
    history_df.plot(title='Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.grid(True)
    plt.show()
    print("Plot displayed.")

if __name__ == '__main__':
    main()
