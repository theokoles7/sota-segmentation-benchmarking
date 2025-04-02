"""CellMap Challenge base dataset loading."""

from logging            import Logger

from numpy              import expand_dims, ndarray, repeat, squeeze
from torch              import tensor, Tensor
from torch.utils.data   import Dataset
from zarr               import Group, open

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
        self.__logger__:        Logger =    LOGGER.getChild("cellmap")
        
        # Open dataset.
        self._root_ :           Group =     open(path, mode = "r")

        # Get reconstruction group names.
        self._reconstructions_: list =      list(self._root_.keys())

        # Initialize samples list.
        self._samples_:         list =      []

        # For each reconstruction group...
        for group in self._reconstructions_:

            # Get group from list.
            reconstruction_group:   list =  self._root_[group]

            # Skip if group does not contain "em" volume or labels.
            if "em" not in reconstruction_group or "labels" not in reconstruction_group: continue

            # Initialize EM volume dictionary.
            em_volumes:             dict =  {}
            
            # For each volume in EM...
            for volume_name in reconstruction_group["em"].keys():
                
                # For each group in volume...
                for em_group in reconstruction_group["em"][volume_name]:
                    
                    # Update EM volume dictionary.
                    em_volumes[volume_name] =   em_group[]

    def __getitem__(self,
        idx:    int
    ) -> dict:
        """# Access dataset sample.

        ## Args:
            * idx (int):    Index of sample within dataset.

        ## Returns:
            * dict:
                * image             (Tensor):   Image sample.
                * mask              (Tensor):   Segmentation mask.
                * reconstruction    (str):      Reconstruction group from which sample is taken.
                * em_volume         (str):      EM volume from which sample is taken.
                * gt_volume         (str):      GT volume from which sample is taken.
                * slice_idx         (int):      Slice index of sample.
        """
        # Record sample index.
        sample_idx: int =   self.indices[idx]

        # Extract sample.
        sample:     any =   self._samples_[sample_idx]

        # Extract slice.
        slice_idx:  int =   sample["slide_idx"]

        # Extract slices based on axis.
        if self.slice_axis == 0:    image, mask = sample["em_array"][slice_idx], sample["gt_array"][slice_idx]
        elif self.slice_axis == 1:  image, mask = sample["em_array"][:, slice_idx], sample["gt_array"][:, slice_idx]
        elif self.slice_axis == 2:  image, mask = sample["em_array"][:, :, slice_idx], sample["gt_array"][:, :, slice_idx]

        # If image only has two dimensions...
        if image.ndim == 2:

            # Expand to 3.
            image:  ndarray =   expand_dims(image, axis = -1)

            # Repeat to make 3 channels.
            image:  Tensor =    tensor(repeat(image, repeats = 3, axis = -1))

        # If mask is more than two dimensions...
        if mask.ndim > 2:

            # Remove singletons.
            mask:   Tensor =    tensor(squeeze(mask))

        # Return sample & specifications.
        return {
            "image":            image,
            "mask":             mask,
            "reconstruction":   sample["reconstruction"],
            "em_volume":        sample["em_volume"],
            "gt_volume":        sample["gt_volume"],
            "slice_idx":        slice_idx
        }

    def __len__(self) -> int:
        """# Indicate length of dataset.

        ## Returns:
            * int:  Number of samples in dataset.
        """
        # Provide number of samples.
        return len(self.indices)