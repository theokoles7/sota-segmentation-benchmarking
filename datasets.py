"""Datasets module."""

from logging                import Logger

from numpy                  import array
from torch                  import as_tensor, int64
from torch.utils.data       import DataLoader
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
from torchvision.datasets   import OxfordIIITPet, VOCSegmentation

from utilities              import LOGGER

def load_dataset(
    dataset_name:   str,
    batch_size:     int =   8, 
    num_workers:    int =   4,
    input_size:     tuple = (512, 512),
    **kwargs    
) -> DataLoader:
    """Initialize dataset and loaders.
    
    ## Args:
        * dataset_name  (str, optional):        Dataset on which model(s) will be evaluated. 
                                                Options are "pets", "coco", and "voc". Defaults 
                                                to "pets".
        * batch_size    (int, optional):        Dataloader batch size. Defaults to 8.
        * num_workers   (int, optional):        Number of threads to use for data loading. 
                                                Defaults to 4.
        * input_size    (tuple[int], optional): Clip size for samples. Defaults to (512, 512).
    
    ## Returns:
        * Dataloader:   Initialized data loader.
    """
    # Initialize logger.
    _logger_:               Logger =            LOGGER.getChild("dataset-loader")
    
    # Define image transforms.
    transform:              Compose =           Compose([
                                                    Resize(size = input_size, antialias = True),
                                                    ToTensor(),
                                                    Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                                                ])
    
    # For masks, we need to be careful to preserve the class values.
    # Do NOT normalize masks - just resize and convert to tensor.
    target_transform:       Compose =           Compose([
                                                    Resize(size = input_size, interpolation = InterpolationMode.NEAREST),
                                                    lambda x: as_tensor(array(x), dtype = int64)
                                                ])
    
    # Match dataset selection.
    match dataset_name:
        
        # Oxford IIIT Pets
        case "pets":
            
            # Log action.
            _logger_.info("Loading Oxford-IIIT Pet dataset...")
            
            # Download/verify dataset.
            dataset:        OxfordIIITPet =     OxfordIIITPet(
                                                    root =              "data",
                                                    split =             "test",
                                                    target_types =      "segmentation",
                                                    download =          True,
                                                    transform =         transform,
                                                    target_transform =  target_transform
                                                )
            
            # Define number of classes.
            num_classes:    int =               3
            
        # Pascal VOC Segmentation.
        case "voc":
            
            # Log action.
            _logger_.info("Loading Pascal VOC Segmentation dataset...")
            
            # Download/verify dataset.
            dataset:        VOCSegmentation =   VOCSegmentation(
                                                    root =              "data",
                                                    year =              "2012",
                                                    image_set =         "val",
                                                    download =          True,
                                                    transform =         transform,
                                                    target_transform =  target_transform
                                                )
            # Define number of classes.
            num_classes:    int =               21
            
        # Invalid selection.
        case _: raise ValueError(f"Invalid dataset selection: {dataset_name}")
        
    # Return dataloader.
    return  DataLoader(
                dataset =           dataset,
                batch_size =        batch_size,
                shuffle =           False,
                num_workers =       num_workers
            )