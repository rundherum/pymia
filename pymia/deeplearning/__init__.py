try:
    import tensorflow
except ImportError:
    raise ImportError('Module "tensorflow" not found. Install "tensorflow<2.0.0" to use the deeplearning module.')

try:
    import torch
except ImportError:
    raise ImportError('Module "torch" not found. Install "torch==1.3.1" to use the deeplearning module.')

try:
    import tensorboardX
except ImportError:
    raise ImportError('Module "tensorboardX" not found. Install "tensorboardX==1.9" to use the deeplearning module.')
