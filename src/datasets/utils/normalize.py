import numpy as np

def normalize(tensor: np.ndarray):
    """Normalize function for a single tensor.

    Args:
        block (np.ndarray): input tensor
    Returns:
        np.ndarray: normalized tensor

    """
    if len(tensor.shape) < 4:
        tensor = np.expand_dims(tensor, axis=2)
    mean = np.array([tensor[..., chn, :].mean() for chn in range(tensor.shape[2])])
    std = np.array([tensor[..., chn, :].std() for chn in range(tensor.shape[2])])
    return (tensor - mean[:, np.newaxis]) / std[:, np.newaxis]
