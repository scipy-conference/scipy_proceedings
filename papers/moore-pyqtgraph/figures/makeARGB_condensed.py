import cupy as cp
import numpy as np

def makeARGB(data, lut=None, levels=None, scale=None, useRGBA=False, output=None):
    # condensed variant, full code at:
    # https://github.com/pyqtgraph/pyqtgraph/blob/pyqtgraph-0.12.0/pyqtgraph/functions.py#L1102-L1331
    xp = cp.get_array_module(data) if cp else np

    nanMask = None
    if data.dtype.kind == "f" and xp.isnan(data.min()):
        nanMask = xp.isnan(data)
    # Scaling
    if isinstance(levels, xp.ndarray) and levels.ndim == 2: # rescale each channel independently
        newData = xp.empty(data.shape, dtype=int)
        for i in range(data.shape[-1]):
            minVal, maxVal = levels[i]
            if minVal == maxVal:
                maxVal = xp.nextafter(maxVal, 2 * maxVal)
            rng = maxVal - minVal
            rng = 1 if rng == 0 else rng
            newData[..., i] = (data[..., i] - minVal) * (scale / rng)
        data = newData
    else:
        minVal, maxVal = levels
        rng = maxVal - minVal
        data = (data - minVal) * (scale / rng)
    # LUT
    if xp == cp: # cupy.take only supports "wrap" mode
        data = cp.take(lut, cp.clip(data, 0, lut.shape[0] - 1), axis=0)
    else:
        data = np.take(lut, data, axis=0, mode='clip')

    imgData = output
    if useRGBA:
        order = [0, 1, 2, 3]  # array comes out RGBA
    else:
        order = [2, 1, 0, 3]  # channels line up as BGR in the final image.
    # attempt to use library function to copy data into image array
    fastpath_success = try_fastpath_argb(xp, data, imgData, useRGBA)
    if fastpath_success:
        pass
    elif data.ndim == 2:
        for i in range(3):
            imgData[..., i] = data
    elif data.shape[2] == 1:
        for i in range(3):
            imgData[..., i] = data[..., 0]
    else:
        for i in range(0, data.shape[2]):
            imgData[..., i] = data[..., order[i]]
    if data.ndim != 3 or data.shape[2] != 4:
        imgData[..., 3] = 255
    # apply nan-mask through alpha channel
    if nanMask is not None:
        if xp == cp: # Workaround for https://github.com/cupy/cupy/issues/4693
            imgData[nanMask, :, 3] = 0
        else:
            imgData[nanMask, 3] = 0
    return imgData
