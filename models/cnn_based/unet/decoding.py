"""# Decoder implementation for U-Net model."""

__all__ = ["Decoder"]

from logging                                import Logger

from torch                                  import cat, Tensor
from torch.nn                               import ConvTranspose2d, Module, ModuleList
from torchvision.transforms                 import CenterCrop

from models.cnn_based.unet.convolution      import ConvolutionalBlock
from utilities                              import LOGGER

class Decoder(Module):
    """# U-Net Decoder class."""
    
    def __init__(self,
        dimensions: tuple[int] =    (64, 32, 16),
        **kwargs
    ):
        """# Initialize Decoder layer.

        ## Args:
            * dimensions    (tuple[int], optional): Dimensions of layers, such that the first 
                                                    element in the tuple is the input dimension of 
                                                    the first layer, and the second element is the 
                                                    output of the first layer and input of the 
                                                    second layer, etc. Defaults to (64, 32, 16).
        """
        # Initialize module.
        super(Decoder, self).__init__()
        
        # Initialize logger.
        self.__logger__:        Logger =        LOGGER.getChild(f"unet-decoder")
        
        # Record dimensions.
        self._dimensions_:      tuple[int] =    dimensions
        
        # Define upsampling layers.
        self._upsampling_:      ModuleList =    ModuleList([
                                                    ConvTranspose2d(
                                                        in_channels =   dimensions[d],
                                                        out_channels =  dimensions[d + 1],
                                                        kernel_size =   2,
                                                        stride =        2
                                                    ) for d in range(len(dimensions) - 1)
                                                ])
        
        # Define decoder blocks.
        self._decoder_blocks_:  ModuleList =    ModuleList([
                                                    ConvolutionalBlock(
                                                        channels_in =   dimensions[d],
                                                        channels_out =  dimensions[d + 1]
                                                    ) for d in range(len(dimensions) - 1)
                                                ])
        
    def forward(self,
        X:                  Tensor,
        encoding_features:  list[Tensor]
    ) -> Tensor:
        """# Forward pass through network.

        ## Args:
            * X                 (Tensor):       Input tensor.
            * encoding_features (list[Tensor]): List of tensors from encoder.

        ## Returns:
            * Tensor:   Output tensor.
        """
        # For each decoder block, upsampling layer, encoding feature, and decoder block...
        for (
                decoder_block,
                upsampling,
                encoder_feature,
                decoding_block
            ) in zip(self._decoder_blocks_, self._upsampling_, encoding_features, self._decoder_blocks_):
            
            # Forward pass through upsampling layer.
            X:                  Tensor =    upsampling(X)
            
            # Crop sample.
            encoding_feature:   Tensor =    self.crop(
                                                encoded_features =  encoding_feature,
                                                X =                 X
                                            )
            
            # Concatenate tensors.
            X:                  Tensor =    cat(tensors = [X, encoding_feature], dim = 1)
            
            # Decode feature.
            X:                  Tensor =    decoder_block(X)
            
        # Return final output.
        return X
            
    def crop(self,
        encoded_features:   Tensor,
        X:                  Tensor
    ) -> any:
        """# Center crop image sample.

        ## Args:
            * encoded_features  (Tensor):   Encoder feature.
            * X                 (Tensor):   Input tensor.

        ## Returns:
            * any:  Cropped feature.
        """
        # Record shape of input tensor.
        (_, _, H, W) =              X.shape
        
        # Return cropped features.
        return CenterCrop([H, W])(encoded_features)
        
        