"""Automation for downloading CellMap challenge dataset."""

from logging    import Logger
from os         import makedirs
from os.path    import exists
from subprocess import PIPE, Popen

from git        import Repo

from utilities  import LOGGER

def download_dataset(
    repository_destination: str =   "..",
    dataset_destination:    str =   "data",
    access_mode:            str =   "append",
    batch_size:             int =   256,
    num_workers:            int =   32,
    raw_padding:            int =   0,
    **kwargs
) -> str:
    """# Download CellMap Challenge dataset.

    ## Args:
        * repository_destination    (str, optional):    Directory under which CellMap Chellenge 
                                                        repository can be cloned. Defaults to parent 
                                                        directory..
        * dataset_destination       (str, optional):    Directory under which dataset will be 
                                                        downloaded. Defaults to "./data/".
        * access_mode               (str, optional):    Access mode for downloading data. Defaults 
                                                        to "append". append = "No error if data 
                                                        already exists". overwrite = "Overwrites 
                                                        existing download(s)".
        * batch_size                (int, optional):    Number of files to fetch in each batch. 
                                                        Defaults to 256.
        * num_workers               (int, optional):    Number of workers to use for parallel 
                                                        download. Defaults to 32.
        * raw_padding               (int, optional):    Padding to apply to raw data, in voxels. 
                                                        Defaults to 0.

    ## Returns:
        * str:  Directory under which dataset was downloaded.
    """
    # Initialize logger.
    __logger__:     Logger =    LOGGER.getChild("download-dataset")
    
    # Log action.
    __logger__.info(f"Downloading CellMap Cahllenge dataset ({locals()})")
    
    # Ensure that dataset destination exists.
    makedirs(name = dataset_destination, exist_ok = True)
    
    try:# If repository has not already been cloned...
        if not exists(f"{repository_destination}/cellmap-segmentation-challenge"):
        
            # Clone cellmap-segmentation-challenge repository.
            cloned_repository:  Repo =  Repo.clone_from(
                                            url =       "git@github.com:janelia-cellmap/cellmap-segmentation-challenge.git",
                                            to_path =   f"{repository_destination}/cellmap-segmentation-challenge"
                                        )
        
    # Catch wildcard errors.
    except Exception as e:  __logger__.critical(f"Failed to clone repository: {e}", exc_info = True)
        
    try:# Initiate installation of repository.
        installation_process:   Popen = Popen(
                args =      ["pip", "install", f"{repository_destination}/cellmap-segmentation-challenge"],
                stderr =    PIPE
            )
        
        # Capture output for debugging.
        stdout, stderr =    installation_process.communicate()
        
    # Catch wildcard errors.
    except Exception as e:  __logger__.critical(f"Failed to install repository: {e}", exc_info = True)
    
    try:# Download dataset.
    
        download_process:       Popen = Popen(
            args =  ["csc", "fetch-data", "-d", dataset_destination, "-m", access_mode, "-b", str(batch_size), "-w", str(num_workers), "-p", str(raw_padding)],
            stderr =    PIPE
        )
        
        # Capture output for debugging.
        stdout, stderr =    download_process.communicate()
        
        # Determine result of command.
        match download_process.returncode:
            
            # Success.
            case 0: __logger__.info(f"Successfully downloaded CellMap Challenge dataset.")
            
            # Otherwise, log error for debugging.
            case _: __logger__.error(f"Failed to download dataset:\n{stderr}")
        
    # Catch wildcard errors.
    except Exception as e:  __logger__.critical(f"Failed to download dataset: {stderr}", exc_info = True)