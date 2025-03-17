"""CellMap Challenge base dataset loading."""

from logging            import Logger

from torch.utils.data   import Dataset

from utilities          import LOGGER

class CellMapDataset(Dataset):
    """# CellMap Challenge dataset loader interface."""
    
    def __init__(self,
        path:               str,
        mode:               str =   "train",
        validation_split:   float = 0.15,
        test_split:         float = 0.15,
        seed:               int =   42,
        **kwargs
    ):
        """# Initialize CellMap Challenge dataset.

        ## Args:
            * path              (str):              Path from which .zarr file will be loaded.
            * mode              (str, optional):    Choice of "train", "validation", or "test". 
                                                    Defaults to "train".
            * validation_split  (float, optional):  Portion of data to use for validation. Defaults 
                                                    to 0.15.
            * test_split        (float, optional):  Portion of data to use for testing. Defaults to 
                                                    0.15.
            * seed              (int, optional):    Random seed for reproducibility. Defaults to 42.
        """
        # Initialize logger.
        self.__logger__:    Logger =    LOGGER.getChild("cellmap")
        
        