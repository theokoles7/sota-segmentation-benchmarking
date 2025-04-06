"""Segmentation models package."""

__all__ = ["DeepLabV3", "MedSAM", "SAM", "Seg_Former", "Swin_Unet", "UNet", "load_model"]

from models.cnn_based           import DeepLabV3, UNet
from models.foundational        import MedSAM, SAM
from models.transformer_based   import Seg_Former, Swin_Unet

def load_model(
    model:  str,
    **kwargs
) -> object:
    """# Load a segmentation model by name.

    ## Args:
        * model_name    (str):  Name of the model to load.

    ## Returns:
        * object:   Loaded segmentation model.
    """
    # Match model selection.
    match model:
        
        # DeepLabV3+
        case "deeplabv3":   return  DeepLabV3(**kwargs)
        
        # U-Net
        case "unet":        return  UNet(
                                        channels_out =          kwargs["channels_out"],
                                        encoder_channels =      kwargs["encoder_channels"],
                                        decoder_channels =      kwargs["decoder_channels"],
                                        segmentation_classes =  kwargs["segmentation_classes"],
                                        retain_dimension =      kwargs["retain_dimension"],
                                    )