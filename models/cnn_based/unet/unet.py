"""U-Net model implementation.

Based on the 2015 paper by Olaf Ronneberger et al.:
https://arxiv.org/pdf/1505.04597

Informed by sources:
* Geeks4Geeks: https://www.geeksforgeeks.org/u-net-architecture-explained/#
"""

__all__ = ["UNet"]

from logging                            import Logger

from torch                              import Tensor
from torch.nn                           import Conv2d, Module
from torch.nn.functional                import interpolate

from models.cnn_based.unet.decoding     import Decoder
from models.cnn_based.unet.encoding     import Encoder
from utilities                          import LOGGER

class UNet(Module):
    """U-Net (CNN-based) segmentation model."""
    
    def __init__(self,
        channels_out:           tuple[int],
        encoder_channels:       tuple[int] =    (3, 16, 32, 64),
        decoder_channels:       tuple[int] =    (64, 32, 16),
        segmentation_classes:   int =           1,
        retain_dimension:       bool =          True,
        **kwargs
    ):
        """# Initialize U-Net model.

        ## Args:
            * channels_out          (tuple[int]):           Determines the spatial dimensions of the 
                                                            output segmentation map. We set this to 
                                                            the same dimension as our input image.
            * encoder_channels      (tuple[int], optional): Defines the gradual increase in channel 
                                                            dimension as our input passes through 
                                                            the encoder. We start with 3 channels 
                                                            (i.e., RGB) and subsequently double the 
                                                            number of channels. Defaults to (3, 16, 
                                                            32, 64).
            * decoder_channels      (tuple[int], optional): Defines the gradual decrease in channel 
                                                            dimension as our input passes through 
                                                            the decoder. We reduce the channels by a 
                                                            factor of 2 at every step. Defaults to 
                                                            (64, 32, 16).
            * segmentation_classes  (int, optional):        Defines the number of segmentation 
                                                            classes where we have to classify each 
                                                            pixel. This usually corresponds to the 
                                                            number of channels in our output 
                                                            segmentation map, where we have one 
                                                            channel for each class. Defaults to 1.
            * retain_dimension      (bool, optional):       Indicates whether we want to retain the 
                                                            original output dimension. Defaults to 
                                                            True.
        """
        # Initialize module.
        super(UNet, self).__init__()
        
        # Initialize logger
        self.__logger__:            Logger =        LOGGER.getChild("u-net")
        
        # Initialize encoder.
        self._encoder_:             Encoder =       Encoder(dimensions =    encoder_channels)
        
        # Initialize decoder.
        self._decoder_:             Decoder =       Decoder(dimensions =    decoder_channels)
        
        # Initialize regression head.
        self._head_:                Conv2d =        Conv2d(
                                                        in_channels =   decoder_channels[-1],
                                                        out_channels =  segmentation_classes,
                                                        kernel_size =   1
                                                    )
        
        # Define attributes.
        self._retain_dimension_:    bool =          retain_dimension
        self._channels_out_:        tuple[int] =    channels_out
        
        # Log initialization for debugging.
        self.__logger__.debug(f"Initialized U-Net model ({locals()})")
        
    def forward(self,
        X:  Tensor
    ) -> Tensor:
        """# Forward pass through network.

        ## Args:
            * X (Tensor):   Input tensor.

        ## Returns:
            * Tensor:   Output tensor.
        """
        # Encode features.
        encoded_features:   list[Tensor] =  self._encoder_(X)
        
        # Decode features.
        decoded_features:   Tensor =        self._decoder_(
                                                X =                 encoded_features[::-1][0],
                                                encoding_features = encoded_features[::-1][1:]
                                            )
        
        # Obtain segmentation mask.
        map:                Tensor =        self._head_(decoded_features)
        
        # Interploate output dimensions.
        if self._retain_dimension_: interpolate(input = map, size = self._channels_out_)
        
        # Return segmentation map.
        return map