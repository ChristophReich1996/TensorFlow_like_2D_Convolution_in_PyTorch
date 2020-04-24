import torch
import torch.nn as nn
import torch.nn.functional as F

class TFConv2d(nn.Module):
    """
    This module implements a TensorFlow like convolution, where the number of input channels gets determend if
    forward is called for the first time.
    """

    def __init__(self, filters: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 bias: bool = True) -> None:
        """
        Constructor method
        :param filters: (int) Number of filters
        :param kernel_size: (int) Kernel size to be utilized
        :param stride: (int) Stride factor to be used
        :param padding: (int) Padding factor
        :param bias: (bool) If true bias is used after convolution
        """
        # Call super constructor
        super(TFConv2d, self).__init__()
        # Save arguments
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Init bias weights
        self.bias = nn.Parameter(torch.zeros(filters)) if bias else None
        # Init weight tensor
        self.weight: nn.Parameter = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of shape (batch size, in channels, height, width)
        :return: (torch.Tensor) Output tensor of shape (batch size, out channels, height, width)
        """
        # Check if weights are not none
        if self.weight is None:
            # Init weight with random normal tensor. Number of input channels are determent by input tensor.
            self.weight = nn.Parameter(
                torch.randn(self.filters, input.shape[1], self.kernel_size, self.kernel_size, dtype=torch.float,
                            device=self.bias.device))
        # Perform convolution
        output = F.conv2d(input=input, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        return output
