import torch.nn as nn
# Assuming you have the convolution function defined in py_utils.py
from py_utils import convolution 

def make_pool_layer() -> nn.Module:
    return nn.Sequential()

# Parameters:
# kernel: Size of the convolutional kernel.
# dim0: Dimension of the input layer.
# dim1: Dimension of the output layer.
# mod: Number of additional modules to add, besides the first one.
# layer: Function used to construct the layers, defaults to the convolution function.
# Returns an nn.Module: A PyTorch module.
def make_hg_layer(kernel: int, dim0: int, dim1: int, mod: int, layer=convolution, **kwargs) -> nn.Module:
    # Ensure kernel size, dim0 and dim1 are positive and mod is non-negative
    assert kernel > 0 and dim0 > 0 and dim1 > 0, "Kernel size and dimensions must be positive."
    assert mod >= 0, "Number of modules must be non-negative."
    
    # Create the first layer using the provided layer function (default is convolution).
    # Store this layer in a list called layers with stride=2.
    layers = [layer(kernel, dim0, dim1, stride=2)]
    
    # Add mod - 1 additional layers using the same kernel and dim1 parameters.
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    
    return nn.Sequential(*layers)

# Ensure you run this code in a Python script or an IDE that supports Python execution.
