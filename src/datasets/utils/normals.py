import numpy as np

def normals(depthmap, normalize=True, keep_dims=True):
    """Calculate depth normals as normals = gF(x,y,z) = (-dF/dx, -dF/dy, 1)

    Args:
        depthmap (np.ndarray): depth map of any dtype, single channel, len(depthmap.shape) == 3
        normalize (bool, optional): if True, normals will be normalized to have unit-magnitude
            Default: True
        keep_dims (bool, optional):
            if True, normals shape will be equals to depthmap shape,
            if False, normals shape will be smaller than depthmap shape.
            Default: True

    Returns:
        Depth normals

    """
    depthmap = np.asarray(depthmap, np.float32)

    if keep_dims is True:
        mask = depthmap != 0
    else:
        mask = depthmap[1:-1, 1:-1] != 0

    if keep_dims is True:
        normals = np.zeros((depthmap.shape[0], depthmap.shape[1], 3), dtype=np.float32)
        normals[1:-1, 1:-1, 0] = - (depthmap[2:, 1:-1] - depthmap[:-2, 1:-1]) / 2
        normals[1:-1, 1:-1, 1] = - (depthmap[1:-1, 2:] - depthmap[1:-1, :-2]) / 2
    else:
        normals = np.zeros((depthmap.shape[0] - 2, depthmap.shape[1] - 2, 3), dtype=np.float32)
        normals[:, :, 0] = - (depthmap[2:, 1:-1] - depthmap[:-2, 1:-1]) / 2
        normals[:, :, 1] = - (depthmap[1:-1, 2:] - depthmap[1:-1, :-2]) / 2
    normals[:, :, 2] = 1

    normals[~mask] = [0, 0, 0]

    if normalize:
        div = np.linalg.norm(normals[mask], ord=2, axis=-1, keepdims=True).repeat(3, axis=-1) + 1e-12
        normals[mask] /= div

    return normals


def normals_multi(depthmaps, normalize=True, keep_dims=True):
    """Calculate depth normals for multiple depthmaps inputs

    Args:
        depthmap (np.ndarray): multiple input depth maps
        normalize (bool, optional): if True, normals will be normalized to have unit-magnitude
            Default: True
        keep_dims (bool, optional):
            if True, normals shape will be equals to depthmap shape,
            if False, normals shape will be smaller than depthmap shape.
            Default: True

    Returns:
        Depth normals

    """
    n_out = np.zeros((depthmaps.shape[0], depthmaps.shape[1], 3, depthmaps.shape[-1]))
    for i in range(depthmaps.shape[-1]):
        n_out[..., i] = normals(depthmaps[..., 0, i], normalize, keep_dims)
    return n_out