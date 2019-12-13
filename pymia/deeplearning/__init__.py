try:
    import torch  # because the data handling uses torch anyways
except ImportError:
    raise ImportError('Module "torch" not found. Install "torch==1.3.1" to use the deeplearning module.')

