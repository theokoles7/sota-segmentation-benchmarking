"""Decoder component for U-Net model."""

from logging    import Logger

from torch.nn   import Module

from utilities  import LOGGER

class Decoder(Module):
    """# Decoder component for U-Net model."""
    
    def __init__(self,
        in_channels_skip_connection:    int,
        dimensions:                     int,
        insampling_type:                str,
        num_decoding_blocks:            int,
        normalization:                  str,
        preactivation:                  bool =  False,
        residual:                       bool =  False,
        padding:                        int =   0,
        padding_mode:                   str =   "zeros",
        activation:                     str =   "ReLU",
        initial_dilation:               int =   1,
        dropout:                        float = 0.0,
        **kwargs
    ):
        """# Initialize the U-Net decoder component.
        
        ## Args:
            * in_channels_skip_connection   (int):              Number of channels in the skip 
                                                                connections from the encoder.
            * dimensions                    (int):              Dimensionality of the input (2 for 
                                                                2D images, 3 for 3D volumes).
            * upsampling_type               (str):              Type of upsampling method ('conv' 
                                                                for transposed convolution or 
                                                                'nontrainable' for nearest/linear 
                                                                upsampling).
            * num_decoding_blocks           (int):              Number of decoding blocks in the 
                                                                decoder.
            * normalization                 (str, optional):    Type of normalization to use 
                                                                ('batch', 'instance', 'group', etc.) 
                                                                or None.
            * preactivation                 (bool, optional):   Whether to use preactivation 
                                                                (normalization->activation->
                                                                convolution). Defaults to False.
            * residual                      (bool, optional):   Whether to use residual connections 
                                                                within each block. Defaults to False.
            * padding                       (int, optional):    Amount of padding to apply in 
                                                                convolutions. Defaults to 0.
            * padding_mode                  (str, optional):    Type of padding ('zeros', 'reflect', 
                                                                'replicate', or 'circular'). 
                                                                Defaults to "zeros".
            * activation                    (str, optional):    Type of activation function to use. 
                                                                Defaults to "ReLU".
            * initial_dilation              (int, optional):    Initial dilation value for dilated 
                                                                convolutions. Defaults to None.
            * dropout                       (float, optional):  Dropout probability. Defaults to 0.
        """
        # Initialize module.
        super(Decoder, self).__init__()
        
        # Initialize logger.
        self.__logger__:    Logger =    LOGGER.getChild("unet.decoder")
        
        