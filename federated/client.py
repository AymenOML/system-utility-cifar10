import tensorflow as tf

def make_tf_dataset(x, y, batch_size=32):
    return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x)).batch(batch_size)

def split_data_among_clients(x, y, num_clients):
    client_data = []
    samples_per_client = len(x) // num_clients

    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client
        client_dataset = make_tf_dataset(x[start:end], y[start:end])
        client_data.append(client_dataset)

    return client_data
