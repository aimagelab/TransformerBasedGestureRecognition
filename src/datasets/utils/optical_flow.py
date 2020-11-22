import numpy as np
import cv2

def dense_flow(clip, rgb=True):
    """Calculate optical flow with Farneback algorithm

    Args:
        clip: input video clip
        rgb: if True, it will covert to gray level every frames
            Default: True

    Returns:
        flow: Calculated Optical flow

    """
    prev = clip[..., 0]
    if rgb:
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flow = np.zeros((clip.shape[0], clip.shape[1], 2, clip.shape[-1] - 1))
    for i in range(1 ,clip.shape[-1]):
        next = clip[..., i]
        if rgb:
            next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
        flow_calc = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow[..., i - 1] = flow_calc
        prev = next
    return flow