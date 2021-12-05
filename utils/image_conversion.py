import numpy as np
import base64


def array_to_base64(arr):
    """convert numpy array to base64 str

    Args:
        arr (np.array): img array
    Returns:
        str: img array base64 string
    """
    # arr = arr.tobytes()
    arr_base64 = base64.b64encode(arr)
    return arr_base64


def base64_toarr(arr_str, shape=28):
    """convert base64 string to nump array(img)

    Args:
        arr_str (str): base64 img string

    Returns:
        np.array: image array
    """
    arr = base64.decodebytes(arr_str)
    np_arr = np.frombuffer(arr, dtype=np.float32)
    np_arr = np_arr.reshape(shape, shape)
    np_arr = np.expand_dims(np_arr, axis=-1)
    return np_arr
