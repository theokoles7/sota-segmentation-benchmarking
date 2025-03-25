"""Encoder for U-Net model."""

__all__ = ["Encoder"]

from logging                                import Logger

from torch                                  import Tensor
from torch.nn                               import MaxPool2d, Module, ModuleList

from models.cnn_based.unet.convolution      import ConvolutionalBlock
from utilities                              import LOGGER

class Encoder(Module):
    """# U-Net Encoder class."""
    
    def __init__(self,
        dimensions: tuple[int] =    (3, 16, 32, 64),
        **kwargs
    ):
        """# Initialize Encoder layer.

        ## Args:
            * dimensions    (tuple[int], optional): Dimensions of layers, such that the first 
                                                    element in the tuple is the input dimension of 
                                                    the first layer, and the second element is the 
                                                    output of the first layer and input of the 
                                                    second layer, etc. Defaults to (3, 16, 32, 64).
        """
        # Initialize module.
        super(Encoder, self).__init__()
        
        # Initialize logger.
        self.__logger__:        Logger =        LOGGER.getChild(f"unet-encoder")
        
        # Define encoder blocks.
        self._encoder_blocks_:  ModuleList =    ModuleList([
                                                    ConvolutionalBlock(
                                                        channels_in =   dimensions[d],
                                                        channels_out =  dimensions[d + 1],
                                                    ) for d in range(len(dimensions) - 1)
                                                ])
        
        # Define pooling layer.
        self._pool_:            MaxPool2d =     MaxPool2d(kernel_size = 2)
        
    def forward(self,
        X:  Tensor
    ) -> list[Tensor]:
        """# Forward pass input through network.

        ## Args:
            * X (Tensor):   Input tensor.

        ## Returns:
            * list[Tensor]: List of output tensors from blocks.
        """
        # Initialize block outputs list.
        block_outputs:  list =  []
        
        # For each block defined...
        for block in self._encoder_blocks_:
            
            # Pass through block.
            X:  Tensor =    block(X)
            
            # Append to list.
            block_outputs.append(X)
            
            # Administer pooling for next layer.
            X:  Tensor =    self._pool_(X)
            
        # Return block outputs.
        return block_outputs