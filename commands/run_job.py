"""Job process execution."""

from logging    import Logger

from datasets   import load_dataset
from models     import *
from utilities  import LOGGER

def run_job(
    datasplit:  str,
    model:      str,
    **kwargs
) -> dict:
    """# Run a job with the specified dataset and model.

    ## Args:
        * datasplit (int):  Datasplit on which model will be trained & evaluated.
        * model     (str):  Model to be used for training & evaluation.

    ## Returns:
        * dict: A dictionary containing the status of the job execution.
    """
    # Initialize logger.
    __logger__:             Logger = LOGGER.getChild("job-process")
    
    # Log action.
    __logger__.info(f"Initializing job process ({locals()})")
    
    # Load the dataset.
    train_data, test_data = load_dataset(datasplit = datasplit)

    # Initialize the model.
    model =                 load_model(
                                model =         model,
                                **kwargs
                            )
    
    