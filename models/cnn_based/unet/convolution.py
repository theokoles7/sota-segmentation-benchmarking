"""Convolutional component for U-Net model.

Implementation based on: https://github.com/fepegar/unet/blob/main/unet/conv.py
"""

from logging    import Logger

from torch      import nn, Tensor
from torch.nn   import Module, ModuleList, Sequential

from utilities  import LOGGER

class ConvolutionalBlock(Module):
    """Convolutional block component for U-Net model."""
    
    def __init__(self,
        dimensions:         int,
        in_channels:        int,
        out_channels:       int,
        normalization:      str =   None,
        kernel_size:        int =   3,
        activation:         str =   "ReLU",
        preactivation:      bool =  False,
        padding:            int =   0,
        padding_mode:       str =   "zeros",
        dilation:           int =   1,
        dropout:            float = 0.0,
        **kwargs
    ):
        """# Initialize convolutional block layer.

        ## Args:
            * dimensions    (int):              Dimensions of input (1, 2, 3, etc.).
            * in_channels   (int):              Number of input channels.
            * out_channels  (int):              Number of output channels.
            * normalization (str, optional):    Normalization method. Defaults to None.
            * kernel_size   (int, optional):    Kernel size (square). Defaults to 3.
            * activation    (str, optional):    Activation function. Defaults to "ReLU".
            * preactivation (bool, optional):   Enable preactivation. Defaults to False.
            * padding       (int, optional):    Padding value. Defaults to 0.
            * padding_mode  (str, optional):    Padding mode. Defaults to "zeros".
            * dilation      (int, optional):    Enable dilation. Defaults to 1.
            * dropout       (float, optional):  Dropout rate. Defaults to 0.0.
        """
        # Initialize module.
        super(ConvolutionalBlock, self).__init__()
        
        # Initialize logger.
        self.__logger__:        Logger =            LOGGER.getChild("unet.conv")
        
        # Initialize module block.
        block:                  ModuleList =        ModuleList()
        
        # Calculate padding.
        padding:                int =               (kernel_size + 2 * (dilation - 1) - 1) // 2
        
        # Define convolutional layer.
        self._convolutional_layer_: any =           getattr(nn, f"Conv{dimensions}d")(
                                                        in_channels =   in_channels,
                                                        out_channels =  out_channels,
                                                        kernel_size =   kernel_size,
                                                        padding =       padding,
                                                        padding_mode =  padding_mode,
                                                        dilation =      dilation,
                                                        bias =          (not preactivation and (normalization is not None))
                                                    )
        
        # Define normalization layer.
        self._normalization_layer_: any =           getattr(nn, f"{normalization.upper()}Norm{dimensions}d")(
                                                        num_features =  (in_channels if preactivation else out_channels)
                                                    ) if normalization is not None else None
        
        # Define activation layer.
        self._activation_layer_:    any =           getattr(nn, activation)() if activation is not None else None
            
        # Define dropout layer.
        self._dropout_layer_:       any =           getattr(nn, f"Dropout{dimensions}d")(p = dropout)
        
        # Build block.
        if preactivation:
            self.add_if_not_none(module_list =  block,  module =    self._normalization_layer_)
            self.add_if_not_none(module_list =  block,  module =    self._activation_layer_)
            self.add_if_not_none(module_list =  block,  module =    self._convolutional_layer_)
            self.add_if_not_none(module_list =  block,  module =    self._dropout_layer_)
        else:
            self.add_if_not_none(module_list =  block,  module =    self._convolutional_layer_)
            self.add_if_not_none(module_list =  block,  module =    self._normalization_layer_)
            self.add_if_not_none(module_list =  block,  module =    self._activation_layer_)
            self.add_if_not_none(module_list =  block,  module =    self._dropout_layer_)
            
        # Define block.
        self._block_:               Sequential =    Sequential(*block)
        
        # Log initialization for debugging.
        self.__logger__.debug(f"ConvolutionBlock initialized ({locals()})")
        
    def forward(self, X: Tensor) -> Tensor:
        """# Forward pass input through network.

        ## Args:
            * X (Tensor):   Input tensor.

        ## Returns:
            * Tensor:   Output tensor.
        """
        # Log input shape for debugging.
        self.__logger__.debug(f"Input shape: {X.shape}")
        
        # Return processed data.
        return self._block_(X)

    @staticmethod
    def add_if_not_none(
        module_list:    ModuleList,
        module:         Module
    ):
        """# Add module to list.

        ## Args:
            * module_list   (ModuleList):   Module list being built.
            * module        (Module):       Module being added to list.
        """
        # Add Module to Module list if Module is not None.
        if module is not None: module_list.append(module)