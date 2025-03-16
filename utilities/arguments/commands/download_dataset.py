"""Argument definitions for download dataset."""

__all__ = ["add_download_dataset_parser"]

from argparse   import ArgumentParser, _SubParsersAction

def add_download_dataset_parser(
    parent_subparser:   _SubParsersAction
) -> None:
    """# Add parser/arguments for downloading dataset.

    ## Args:
        * parent_subparser  (_SubParsersAction):    Parent's sub-parser.
    """
    # Initialize parser.
    _parser_:   ArgumentParser =    parent_subparser.add_parser(
                                        name =  "download-dataset",
                                        help =  """Download CellMap Challenge dataset. CAUTION: The 
                                                download process will likely take at least an hour."""
                                    )
    
    _parser_.add_argument(
        "--repository_destination",
        type =      str,
        default =   "..",
        help =      f"""Directory under which CellMap Chellenge repository can be cloned. Defaults 
                    to parent directory."""
    )
    
    _parser_.add_argument(
        "--dataset-destination", "-d",
        type =      str,
        default =   "data",
        help =      f"""Directory under which dataset will be downloaded. Defaults to "./data/"."""
    )
    
    _parser_.add_argument(
        "--access-mode", "-m",
        type =      str,
        choices =   ["append", "overwrite"],
        default =   "append",
        help =      f"""Access mode for downloading data. Defaults to "append". append = "No error 
                    if data already exists". overwrite = "Overwrites existing download(s)"."""
    )
    
    _parser_.add_argument(
        "--batch-size", "-b",
        type =      int,
        default =   256,
        help =      """Number of files to fetch in each batch. Defaults to 256."""
    )
    
    _parser_.add_argument(
        "--num-workers", "-w",
        type =      int,
        default =   32,
        help =      """Number of workers to use for parallel download. Defaults to 32."""
    )
    
    _parser_.add_argument(
        "--raw-padding", "-p",
        type =      int,
        default =   0,
        help =      """Padding to apply to raw data, in voxels. Defaults to 0."""
    )