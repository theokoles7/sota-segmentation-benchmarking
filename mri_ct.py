"""MRI/CT scan dataset."""

from logging                    import Logger
from os                         import listdir

from nibabel                    import load
from numpy                      import float32
from torch                      import from_numpy, Tensor
from torch.utils.data           import DataLoader, Dataset

from utilities                  import LOGGER

class MRI_CT(Dataset):
    """MRI/CT scan dataset class."""
    
    def __init__(self,
        data_path:  str =       "data/MRI_CT",
        transform:  callable =  None,
        **kwargs
    ):
        """# Initialize MRI/CT scan dataset.

        ## Args:
            * data_path (str, optional):        Path at which dataset files can be located. Defaults 
                                                to "data/MRI_CT".
            * transform (callable, optional):   Data transform. Defaults to None.
        """
        # Initialize logger.
        self.__logger__:    Logger =    LOGGER.getChild("mri-ct")
        
        # Initialize dataset.
        super(MRI_CT, self).__init__()
        
        # Define attributes.
        self.__root__:      str =       data_path
        
        # Define transform if provided.
        self.__transform__: callable =  transform
        
        # Define files list.
        self.__images__:    list =      listdir(f"{self.__root__}/images")
        self.__labels__:    list =      listdir(f"{self.__root__}/labels")
        
        # Log initialization for debugging.
        self.__logger__.debug(f"MRI/CT scan dataset initialized ({locals()})")
        
    def __len__(self) -> int:
        """# Provide length of dataset.

        ## Returns:
            * int:  Number of dataset samples.
        """
        # Provide number of samples.
        return len(self.__images__)
    
    def __getitme__(self,
        idx:    int
    ) -> dict:
        """# Provide sample from dataset.

        ## Args:
            * idx   (int):  Indexof sample being fetched.

        ## Returns:
            * dict: Dataset sample's image and path.
        """
        # Load Nifti file into Tensor.
        image:  Tensor =    from_numpy(load(f"{self.__root__}/images/{self.__images__[idx]}").get_fdata().astype(float32))
        
        # Add channel dimension if needed.
        if len(image.shape) == 3:           image = image.unsqueeze(0)
        
        # Apply transformation if provided.
        if self.__transform__ is not None:  image = self.__transform__(image)
        
        # Return sample.
        return  {
                    "image":    image,
                    "filename": self.__images__[idx]
                }