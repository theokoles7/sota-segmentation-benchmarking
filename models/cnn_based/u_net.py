"""U-Net model wrapper."""

from logging                        import Logger

from segmentation_models_pytorch    import Unet

from utilities                      import LOGGER

class UNet(Unet):
    """U-Net wrapper class."""
    
    def __init__(self,
        encoder_name:           str =               "resnet34",
        encoder_depth:          int =               5,
        encoder_weights:        str =               "imagenet",
        decoder_user_norm:      str =               "batchnorm",
        decoder_channels:       tuple[int] =        (256, 128, 64, 32, 16),
        decoder_attention_type: str =               None,
        decoder_interpolation:  str =               "nearest",
        in_channels:            int =               3,
        classes:                int =               1,
        activation:             str | Callable =    None,
        aux_params:             dict =              None,
        **kwargs
    ):
        """# Initialize U-Net model.

        ## Args:
            * encoder_name              (str, optional):            Name of the classification model 
                                                                    that will be used as an encoder 
                                                                    (a.k.a backbone) to extract 
                                                                    features of different spatial 
                                                                    resolution. Defaults to 
                                                                    "resnet34".
            * encoder_depth             (int, optional):            A number of stages used in 
                                                                    encoder in range [3, 5]. Each 
                                                                    stage generate features two 
                                                                    times smaller in spatial 
                                                                    dimensions than previous one 
                                                                    (e.g. for depth 0 we will have 
                                                                    features with shapes [(N, C, H, 
                                                                    W),], for depth 1 - [(N, C, H, 
                                                                    W), (N, C, H // 2, W // 2)] and 
                                                                    so on). Defaults to 5.
            * encoder_weights           (str, optional):            One of None (random 
                                                                    initialization), “imagenet” 
                                                                    (pre-training on ImageNet) and 
                                                                    other pretrained weights (see 
                                                                    table with available weights for 
                                                                    each encoder_name). Defaults to 
                                                                    "imagenet".
            * decoder_user_norm         (str, optional):            Specifies normalization between 
                                                                    Conv2D and activation. Accepts 
                                                                    the following types: - True: 
                                                                    Defaults to “batchnorm”. - False: 
                                                                    No normalization (nn.Identity). 
                                                                    - str: Specifies normalization 
                                                                    type using default parameters. 
                                                                    Available values: "batchnorm", 
                                                                    "identity", "layernorm", 
                                                                    "instancenorm", or "inplace". 
                                                                    Defaults to "batchnorm".
            * decoder_channels          (tuple[int], optional):     List of integers which specify 
                                                                    in_channels parameter for 
                                                                    convolutions used in decoder. 
                                                                    Length of the list should be 
                                                                    the same as encoder_depth. 
                                                                    Defaults to (256, 128, 64, 
                                                                    32, 16).
            * decoder_attention_type    (str, optional):            Attention module used in decoder 
                                                                    of the model. Available options 
                                                                    are None and scse. Defaults to 
                                                                    None.
            * decoder_interpolation     (str, optional):            Interpolation mode used in 
                                                                    decoder of the model. Available 
                                                                    options are “nearest”, 
                                                                    “bilinear”, “bicubic”, “area”, 
                                                                    “nearest-exact”. Defaults to 
                                                                    "nearest".
            * in_channels               (int, optional):            A number of input channels for 
                                                                    the model, default is 3 (RGB 
                                                                    images).
            * classes                   (int, optional):            A number of classes for output 
                                                                    mask (or you can think as a 
                                                                    number of channels of output 
                                                                    mask). Defaults to 1.
            * activation                (str | Callable, optional): An activation function to apply 
                                                                    after the final convolution 
                                                                    layer. Available options are 
                                                                    “sigmoid”, “softmax”, 
                                                                    “logsoftmax”, “tanh”, 
                                                                    “identity”. Defaults to None.
            * aux_params                (dict, optional):           Dictionary with parameters of 
                                                                    the auxiliary output 
                                                                    (classification head). 
                                                                    Auxiliary output is build on top 
                                                                    of encoder if aux_params is not 
                                                                    None (default). Defaults to None.
        """
        # Initialize logger.
        self.__logger__:    Logger =    LOGGER.getChild("u-net")
        
        # Initialize model.
        super(UNet, self).__init__(**locals())
        
        # Log initialization for debugging.
        self.__logger__.debug(f"U-Net model initialized ({locals()})")