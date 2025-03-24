"""Decoder component for U-Net model."""

from logging                                import Logger

from torch                                  import Tensor
from torch.nn                               import Module, ModuleList, Upsample

from models.cnn_based.unet.decoding_block   import DecoderBlock
from utilities                              import LOGGER

class Decoder(Module):
    """# Decoder component for U-Net model."""
    
    def __init__(self,
        in_channels_skip_connection:    int,
        dimensions:                     int,
        upsampling_type:                str,
        num_decoding_blocks:            int,
        normalization:                  str =   None,
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
        self.__logger__:        Logger =        LOGGER.getChild("unet.decoder")
        
        # Define attributes.
        self._decoding_blocks:  ModuleList =    ModuleList()
        self.dilation:          int =           initial_dilation
        
        # For each decoding block prescribed...
        for block in range (num_decoding_blocks):
            
            # Append new decoding block.
            self._decoding_blocks.append(
                DecoderBlock(
                    id =                            block,
                    in_channels_skip_connection =   in_channels_skip_connection,
                    dimensions =                    dimensions,
                    upsampling_type =               upsampling_type,
                    normalization =                 normalization,
                    preactivation =                 preactivation,
                    residual =                      residual,
                    padding =                       padding,
                    padding_mode =                  padding_mode,
                    activation =                    activation,
                    dilation =                      self.dilation,
                    dropout =                       dropout
                )
            )
            
            # Update input channels for blocks.
            in_channels_skip_connection //= 2
            
            # Update dilation.
            if self.dilation is not None: self.dilation //= 2
            
    def forward(self,
        skip_connections:   list[Tensor],
        X:                  Tensor
    ) -> Tensor:
        """# Forward pass through U-Net decoder.

        ## Args:
            * skip_connections  (list[Tensor]): List of features maps from the encoder's skip 
                                                connections, ordered from shallowest to deepest. 
                                                These will be used in reverse order during 
                                                decoding.
            * X                 (Tensor):       Input tensor.

        ## Returns:
            * Tensor:   Decoded feature map with the same spatial dimensions as the shallowest 
                        skip connection, ready for final classification.
        """
        # For each connection skip/block...
        for skip_connection, decoding_block in zip(reversed(skip_connections), self.decoding_blocks):
            
            # Record output from each block.
            x = decoding_block(skip_connection, x)
            
        # Return final output.
        return x