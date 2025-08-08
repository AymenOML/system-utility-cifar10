import numpy as np

def serialize_weights(weights):
    """Convert list of tensors to list of NumPy arrays (float32)."""
    return [w.astype(np.float32) for w in weights]

def deserialize_weights(weights):
    """Assumes weights are already NumPy arrays — just pass through."""
    return weights
