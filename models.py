"""Models module."""

from logging                        import Logger

from torch.nn                       import BatchNorm2d, Conv2d, GroupNorm, Module
from segmentation_models_pytorch    import DeepLabV3Plus, FPN, Segformer, Unet
from torch.nn.init                  import constant_, kaiming_normal_

from utilities                      import LOGGER

def load_model(
    model_name:     str,
    num_classes:    int =   3,
    device:         str =   "cuda",
    **kwargs
) -> Module:
    """# Get a segmentation model by name.
    
    ## Args:
        * model_name    (str):  Model selection.
    """
    # Initialize logger.
    _logger_:   Logger =    LOGGER.getChild("model-loader")
    
    # Match model selection.
    match model_name:
        
        # DeepLab-V3+
        case "deeplab-v3":
            
            # Log action.
            _logger_.info("Loading DeepLab-V3+ model.")
            
            # Initialize model.
            model:  DeepLabV3Plus = DeepLabV3Plus(
                                        encoder_name =      "mobilenet_v2",
                                        encoder_weights =   "imagenet",
                                        in_channels =       3,
                                        classes =           num_classes,
                                        activation =        None
                                    )
        
        # FPN
        case "fpn":
            
            # Log action.
            _logger_.info("Loading FPN model.")
            
            # Initialize model.
            model:  FPN =           FPN(
                                        encoder_name =      "mobilenet_v2",
                                        encoder_weights =   "imagenet",
                                        in_channels =       3,
                                        classes =           num_classes,
                                        activation =        None
                                    )
        
        # Seg-Former
        case "seg-former":
            
            # Log action.
            _logger_.info("Loading Seg-Former model.")
            
            # Initialize model.
            model:  Segformer =     Segformer(
                                        encoder_name =      "mobilenet_v2",
                                        encoder_weights =   "imagenet",
                                        in_channels =       3,
                                        classes =           num_classes,
                                        activation =        None
                                    )
        
        # U-Net
        case "u-net":
            
            # Log action.
            _logger_.info("Loading U-Net model.")
            
            # Initialize model.
            model:  Unet =          Unet(
                                        encoder_name =      "mobilenet_v2",
                                        encoder_weights =   "imagenet",
                                        in_channels =       3,
                                        classes =           num_classes,
                                        activation =        None
                                    )
        
        # Invalid selection.
        case _: raise ValueError(f"Invalid model selection: {model_name}")
    
    # For each module within model...
    for m in model.modules():
        
        # Initialize convoling layer weights.
        if isinstance(m, Conv2d):  kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
        
        # Initialize normalization layers.
        elif isinstance(m, (BatchNorm2d, GroupNorm)):
            constant_(m.weight, 1)
            constant_(m.bias, 0)
            
    # Return initialized model, placed on device.
    return model.to(device)