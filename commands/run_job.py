"""Job process execution."""

from datasets   import load_dataset
from models     import *

def run_job(
    datasplit:  str,
    model:      str
) -> dict:
    """# Run a job with the specified dataset and model.

    ## Args:
        * datasplit (int):  Datasplit on which model will be trained & evaluated.
        * model     (str):  Model to be used for training & evaluation.

    ## Returns:
        * dict: A dictionary containing the status of the job execution.
    """
    # Load the dataset.
    train_data, test_data = load_dataset(datasplit = datasplit)

    # Initialize the model.
    model =                 load_model(
                                model_name =    model,
                                num_classes =   len(train_data.classes),
                                in_channels =   3
                            )
    
    