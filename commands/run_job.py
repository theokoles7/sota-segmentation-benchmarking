"""Job process execution."""

from models import *

def run_job(
    dataset:    str,
    model:      str
) -> dict:
    """# Run a job with the specified dataset and model.

    ## Args:
        * dataset   (str):  Dataset on which model will be trained & evaluated.
        * model     (str):  Model to be used for training & evaluation.

    ## Returns:
        * dict: A dictionary containing the status of the job execution.
    """
    # Load the dataset.