from data_loader import load_and_preprocess_data
from federated.client import split_data_among_clients
from federated.server import build_federated_averaging_process

import matplotlib.pyplot as plt

def main():
    print("Loading and partitioning data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    federated_train_data = split_data_among_clients(x_train, y_train, num_clients=5)

    print("Initializing federated training process...")
    iterative_process = build_federated_averaging_process()
    state = iterative_process.initialize()

    print("Starting federated training...")
    rounds = []
    accuracy_list = []
    loss_list = []

    for round_num in range(1, 11):
        state, metrics = iterative_process.next(state, federated_train_data)
        print(f"Round {round_num} metrics: {metrics}")

        rounds.append(round_num)
        accuracy_list.append(metrics['train']['categorical_accuracy'])
        loss_list.append(metrics['train']['loss'])

    print("Federated training complete.")

    # Plot metrics
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, accuracy_list, marker='o')
    plt.title('Federated Training Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(rounds, loss_list, marker='o', color='orange')
    plt.title('Federated Training Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
