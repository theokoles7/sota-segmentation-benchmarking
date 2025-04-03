"""Datasets package."""

__all__ = ["load_dataset"]
    
from cellmap_data.dataloader    import CellMapDataLoader
from cellmap_data.transforms.augment import NaNtoNum, Binarize
from cellmap_segmentation_challenge.utils.dataloader    import get_dataloader
from torch                      import float
from torchvision.transforms.v2 import Compose, ToDtype

def load_dataset(
    datasplit:  int,
    batch_size: int =   1,
    **kwargs
) -> tuple[CellMapDataLoader, CellMapDataLoader]:
    """# Initialize and return data loaders.

    ## Args:
        * datasplit     (int):  Datasplit selection.
        * batch_size    (int):  Data loader batch size. Defaults to 1.

    ## Returns:
        * tuple[CellMapDataLoader, CellMapDataLoader]: Train and validation data loaders.
    """
    # Return initialized loaders.
    return get_dataloader(
        datasplit_path =                f"datasplit{datasplit}.csv",
        classes =                       ["mito", "er"],
        batch_size =                    batch_size,
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