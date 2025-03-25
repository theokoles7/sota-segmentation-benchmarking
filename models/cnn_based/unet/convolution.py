"""# Convolutional block implementation for U-Net model."""

__all__ = ["ConvolutionalBlock"]

from logging    import Logger

from torch      import Tensor
from torch.nn   import Conv2d, Module, ReLU

from utilities  import LOGGER

class ConvolutionalBlock(Module):
    """# U-Net Convolutional Block class."""
    
    def __init__(self,
        channels_in:    int,
        channels_out:   int,
        **kwargs
    ):
        """# Initialize ConvolutionalBlock layer.

        ## Args:
            * channels_in   (int):  Input channels.
            * channels_out  (int):  Output channels.
        """
        # Initialize module.
        super(ConvolutionalBlock, self).__init__()
        
        # Initialize logger.
        self.__logger__:    Logger =    LOGGER.getChild(f"unet-conv-block")
        
        # Define convolving layers.
        self._conv_1_:      Conv2d =    Conv2d(
                                            in_channels =   channels_in,
                                            out_channels =  channels_out,
                                            kernel_size =   3
                                        )
        
        self._conv_2_:      Conv2d =    Conv2d(
                                            in_channels =   channels_out,
                                            out_channels =  channels_out,
                                            kernel_size =   3
                                        )
        
        # Define non-linear layer.
        self._relu_:        ReLU =      ReLU()
        
    def forward(self,
        X:  Tensor
    ) -> Tensor:
        """# Forward pass input through network.

        ## Args:
            * X (Tensor):   Input tensor.

        ## Returns:
            * Tensor:   Output tensor.
        """
        # Apply layers to input and return output.
        return self._conv_2_(self._relu_(self._conv_1_(X)))