import tensorflow_federated as tff
from model import build_cnn_model
import tensorflow as tf
import numpy as np

def get_model_fn():
    # Build a sample batch for input_spec
    sample_batch = tf.data.Dataset.from_tensor_slices(
        (np.zeros((1, 32, 32, 3), dtype=np.float32), np.zeros((1, 10), dtype=np.float32))
    ).batch(1)

    def model_fn():
        keras_model = build_cnn_model()
        return tff.learning.models.from_keras_model(
            keras_model=keras_model,
            input_spec=sample_batch.element_spec,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )

    return model_fn

def build_federated_averaging_process():
    model_fn = get_model_fn()
    return tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )
