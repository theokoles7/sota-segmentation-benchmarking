"""Segmentation models package."""

__all__ = ["DeepLabV3", "MedSAM", "SAM", "Seg_Former", "Swin_Unet", "UNet", "load_model"]

from models.cnn_based           import DeepLabV3, UNet
from models.foundational        import MedSAM, SAM
from models.transformer_based   import Seg_Former, Swin_Unet

def load_model(
    model_name: str,
    pretrained: bool = True,
    num_classes: int = 1,
    in_channels: int = 3,
    **kwargs
) -> object:
    """Load a segmentation model by name.

    Args:
        model_name (str): Name of the model to load.
        pretrained (bool): Whether to load pretrained weights.
        num_classes (int): Number of output classes.
        in_channels (int): Number of input channels.

    Returns:
        object: Loaded segmentation model.
    """
    # Dictionary of available models.
    models = {
        "DeepLabV3":    DeepLabV3,
        "MedSAM":       MedSAM,
        "SAM":          SAM,
        "Seg_Former":   Seg_Former,
        "Swin_Unet":    Swin_Unet,
        "UNet":         UNet
    }
    
    # Raise an error if the model name is not found.
    if model_name not in models: raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")

    return models[model_name](pretrained=pretrained, num_classes=num_classes, in_channels=in_channels, **kwargs
)