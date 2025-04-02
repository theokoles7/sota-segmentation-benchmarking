

from cellmap_data.transforms.augment import NaNtoNum, Binarize
from cellmap_segmentation_challenge.utils.dataloader    import get_dataloader
from torch                      import float
from torchvision.transforms.v2 import Compose, ToDtype

train_loader, val_loader = get_dataloader(
    datasplit_path =                "datasplit.csv",
    classes =                       ["mito", "er"],
    batch_size =                    1,
    input_array_info =              {
                                        "shape": (1, 64, 64),
                                        "scale": (8, 8, 8),
                                    },
    target_array_info =             {
                                        "shape": (1, 64, 64),
                                        "scale": (8, 8, 8),
                                    },
    spatial_transforms =            {
                                        "mirror": {"axes": {"x": 0.5, "y": 0.5}},
                                        "transpose": {"axes": ["x", "y"]},
                                        "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
                                    },
    iterations_per_epoch =          100,
    random_validation =             False,
    device =                        "cuda",
    weighted_sampler =              True,
    use_mutual_exclusion =          False,
    train_raw_value_transforms =    Compose([
                                        ToDtype(dtype = float, scale = True),
                                        NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
                                    ]),
    val_raw_value_transforms =      Compose([
                                        ToDtype(dtype = float, scale = True),
                                        NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
                                    ]),
    target_value_transforms =       Compose([ToDtype(dtype = float), Binarize()]),
)

# print(f"Train loader: {train_loader}")
print(f"Validation loader: {val_loader.classes}")

# """CellMap dataset class."""

# from sklearn.model_selection    import train_test_split
# from torch.utils.data           import Dataset
# from zarr                       import Array, Group, open

# class CellMapDataset(Dataset):
#     """CellMap dataset interface."""
    
#     def __init__(self,
#         zarr_path:      str,
#         images_path:    str =       "em",
#         masks_path:     str =       "labels",
#         transform:      callable =  None,
#         mode:           str =       "train",
#         val_split:      float =     0.15,
#         test_split:     float =     0.15,
#         seed:           int =       42
#     ):
#         """# Initialize CellMap dataset.
        
#         ## Args:
#             * zarr_path     (str):                  Path to the zarr file
#             * images_path   (str):                  Path within zarr to the images array
#             * masks_path    (str):                  Path within zarr to the masks array
#             * transform     (callable, optional):   Optional transform to be applied on a sample
#             * mode          (str):                  'train', 'val', or 'test'
#             * val_split     (float):                Proportion of data to use for validation
#             * test_split    (float):                Proportion of data to use for testing
#             * seed          (int):                  Random seed for reproducibility
#         """
#         # Define transform.
#         self.transform: callable =      transform
        
#         # Record mode.
#         self.mode:      str =           mode
        
#         # Open zarr file
#         self.root:      Array | Group = open(zarr_path, mode='r')
        
#         # Load images and masks directly
#         self.images = self.root[images_path]
#         self.masks = self.root[masks_path]
        
#         # Ensure we have the same number of images and masks
#         assert len(self.images) == len(self.masks), "Number of images and masks must match"
        
#         # Total number of samples
#         num_samples = len(self.images)
        
#         # Create indices for train/val/test split
#         all_indices = list(range(num_samples))
#         train_idx, test_idx = train_test_split(all_indices, test_size=test_split, random_state=seed)
#         train_idx, val_idx = train_test_split(train_idx, test_size=val_split/(1-test_split), random_state=seed)
        
#         # Select indices based on mode
#         if mode == 'train':
#             self.indices = train_idx
#         elif mode == 'val':
#             self.indices = val_idx
#         else:
#             self.indices = test_idx