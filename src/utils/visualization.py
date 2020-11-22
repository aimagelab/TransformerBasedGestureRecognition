import numpy as np
import cv2

def plot_depth(path, depth):
    """Plot a single depth map

    Attributes:
        path (str): Path to save the depth map
        depth (np.ndarray): Depth map data

    """
    if len(depth.shape) > 2:
        if depth.shape[-1] != 1:
            raise ValueError("Wrong number of channel, 1 is required, got {}".format(depth.shape))
        else:
            depth = depth.squeeze()
    tmp = np.zeros((depth.shape[0], depth.shape[1], 3))
    tmp[..., 0] = depth.copy()
    tmp[..., 1] = depth.copy()
    tmp[..., 2] = depth.copy()
    tmp = ((tmp * 255) / tmp.max()).astype(np.uint8)
    cv2.imwrite(path, tmp)
