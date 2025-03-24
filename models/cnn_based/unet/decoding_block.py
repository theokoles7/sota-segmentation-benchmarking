"""DecoderBlock component for U-Net model."""  

from logging                                import Logger

from torch                                  import cat, nn, stack, tensor, Tensor
from torch.nn                               import Module, Upsample
from torch.nn.functional                    import pad

from models.cnn_based.unet.convolution      import ConvolutionalBlock 
from utilities                              import LOGGER

class DecoderBlock(Module):
    def __init__(self,
        id:                             int,
        in_channels_skip_connection:    int,
        dimensions:                     int,
        upsampling_type:                str,
        normalization:                  str =   None,
        preactivation:                  bool =  True,
        residual:                       bool =  False,
        padding:                        int =   0,
        padding_mode:                   str =   "zeros",
        activation:                     str =   "ReLU",
        dilation:                       int =   None,
        dropout:                        float = 0
    ):
        """# Initialize decoder block.
        
        ## Args:
            * in_channels_skip_connection   (int):              Number of channels in the skip 
                                                                connections from the encoder.
            * dimensions                    (int):              Dimensionality of the input (2 for 
                                                                2D images, 3 for 3D volumes).
            * upsampling_type               (str):              Type of upsampling method ('conv' 
                                                                for transposed convolution or 
                                                                'nontrainable' for nearest/linear 
                                                                upsampling).
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
            * dilation                      (int, optional):    Initial dilation value for dilated 
                                                                convolutions. Defaults to None.
            * dropout                       (float, optional):  Dropout probability. Defaults to 0.
        """
        # Initialize module.
        super(DecoderBlock, self).__init__()
        
        # Initialize logger.
        self.__logger__:        Logger =        LOGGER.getChild(f"unet.decoder-block-{id}")
        
        # Record residual decision.
        self.residual:          bool =          residual
        
        # If upsampling type was specified as "convolution"...
        if upsampling_type == "conv":
            
            # Define channels proportional to skip connections.
            channels_in = channels_out = 2 * in_channels_skip_connection
            
            # Define layer.
            self.upsample_layer:        Module =    getattr(nn, f"ConvTranspose{dimensions}d")(in_channels = channels_in, out_channels = channels_out, kernel_size = 2, stride = 2)
            
        # Otherwise, simply create a normal Upsampling layer.
        else:   self.upsample_layer:    Module =    Upsample(scale_factor = 2, mode = upsampling_type, align_corners = False)
        
        # Define convolutional layers.
        self.conv_1:    ConvolutionalBlock =    ConvolutionalBlock(
                                                    dimensions =    dimensions,
                                                    in_channels =   in_channels_skip_connection * 3,
                                                    out_channels =  in_channels_skip_connection,
                                                    normalization = normalization,
                                                    preactivation = preactivation,
                                                    padding =       padding,
                                                    padding_mode =  padding_mode,
                                                    activation =    activation,
                                                    dilation =      dilation,
                                                    dropout =       dropout
                                                )
        
        self.conv_2:    ConvolutionalBlock =    ConvolutionalBlock(
                                                    dimensions =    dimensions,
                                                    in_channels =   in_channels_skip_connection,
                                                    out_channels =  in_channels_skip_connection,
                                                    normalization = normalization,
                                                    preactivation = preactivation,
                                                    padding =       padding,
                                                    padding_mode =  padding_mode,
                                                    activation =    activation,
                                                    dilation =      dilation,
                                                    dropout =       dropout
                                                )
        
        # If residual requested...
        if residual:
            
            # Define residual layer.
            self.conv_residual: ConvolutionalBlock =    ConvolutionalBlock(
                                                            dimensions =    dimensions,
                                                            in_channels =   in_channels_skip_connection * 3,
                                                            out_channels =  in_channels_skip_connection,
                                                            kernel_size =   1,
                                                            normalization = normalization,
                                                            activation =    activation,
                                                        )
            
    def forward(self,
        skip_connection:    Tensor,
        X:                  Tensor
    ) -> Tensor:
        """# Forward pass through U-Net decoder block.

        ## Args:
            * skip_connections  (Tensor):   Features map from the encoder's skip connections.
            * X                 (Tensor):   Input tensor.

        ## Returns:
            * Tensor:   Decoded feature map with the same spatial dimensions as the shallowest 
                        skip connection, ready for final classification.
        """
        # Upsample input.
        X:                  Tensor =    self.upsample_layer(X)
        
        # Create crop.
        skip_connection:    Tensor =    self.center_crop(skip_connection = skip_connection, X = X)
        
        # Concatenate.
        X:                  Tensor =    cat((skip_connection, X), dim = 1)
        
        # User residual layer if requested.
        if self.residual:   return self.conv_2(self.conv_1(X)) + self.conv_residual(X)
        
        # Otherwise, return output from ordinary convolution.
        return self.conv_2(self.conv_1(X))
        
    def center_crop(self,
        skip_connection:    Tensor,
        X:                  Tensor
    ) -> Tensor:
        """# Center crop the skip connection tensor to match the spatial dimensions of x.
    
        In U-Net architecture, the spatial dimensions of skip connections from the encoder
        are typically larger than the upsampled feature maps in the decoder. This method
        crops the skip connection tensor symmetrically from all sides to ensure
        compatible dimensions for concatenation.
        
        ## Args:
            * skip_connection   (Tensor):   Feature map from encoder skip connection with shape 
                                            [batch_size, channels, *spatial_dims].
            * X                 (Tensor):   Upsampled feature map in the decoder with shape 
                                            [batch_size, channels, *spatial_dims].
        
        ## Returns:
            * Tensor:   Center-cropped skip connection tensor with the same spatial dimensions as x, 
                        ready for concatenation.
        
        ## Note:
            This implementation uses negative padding (via F.pad with negative values)
            which effectively crops the tensor.
        """
        # Calculate how much to crop from each side (divide by 2 for center cropping).
        # Convert to integers as tensor indices must be integers.
        half_crop:  Tensor =    ((tensor(skip_connection.shape)[2:] - tensor(X.shape)[2:]) / 2).int()
        
        # We use negative values to indicate cropping rather than padding.
        # The format is (left, right, top, bottom, front, back) or for 2D (left, right, top, bottom).
        # Each dimension gets two values (start, end) which explains the stack and flatten operations.        
        return pad(skip_connection, -stack((half_crop, half_crop)).t().flatten().tolist())