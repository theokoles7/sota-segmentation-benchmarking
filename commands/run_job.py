"""Job process execution."""

from logging        import Logger

from torch          import Tensor
from torch.optim    import RAdam
from torch.nn       import BCEWithLogitsLoss, Module
from tqdm           import tqdm

from datasets       import load_dataset
from models         import *
from utilities      import LOGGER

def run_job(
    datasplit:  str,
    model:      str,
    epochs:     int,
    **kwargs
) -> dict:
    """# Run a job with the specified dataset and model.

    ## Args:
        * datasplit (int):              Datasplit on which model will be trained & evaluated.
        * model     (str):              Model to be used for training & evaluation.
        * epochs    (int, optional):    Epochs for which training phase will execute.

    ## Returns:
        * dict: A dictionary containing the status of the job execution.
    """
    # Initialize logger.
    __logger__:     Logger =    LOGGER.getChild("job-process")
    
    # Log action.
    __logger__.info(f"Initializing job process ({locals()})")
    
    # Load the dataset.
    train_data, test_data =             load_dataset(datasplit = datasplit)

    # Initialize the model.
    model:          Module =            load_model(
                                            model =         model,
                                            **kwargs
                                        )
    
    # Initialize optimizer.
    optimizer:      RAdam =             RAdam(
                                            params =                    model.parameters(),
                                            lr =                        0.01,
                                            decoupled_weight_decay =    True
                                        )
    
    # Define loss function.
    loss_function:  BCEWithLogitsLoss = BCEWithLogitsLoss()
    
    # Initialize job statistics.
    job_statistics: dict =              {
                                            "epochs":           {},
                                            "test_accuracy":    0,
                                            "test_loss":        0
                                        }
    
    # Define input and target keys.
    input_keys:     list =              list(train_data.dataset.input_arrays.keys())
    target_keys:    list =              list(train_data.dataset.target_arrays.keys())
    
    # For each epoch prescribed...
    for epoch in range (1, epochs + 1):
        
        # TRAIN ====================================================================================
        
        # Set model to training mode.
        model.train()

        # Refresh the train loader to shuffle the data yielded by the dataloader.
        train_data.refresh()
        
        # # Initialize loader as iterator.
        # loader: iter =  iter(train_data.loader)
            
        # Reset gradients.
        optimizer.zero_grad()
        
        # Initialize progress bar.
        with tqdm(
            total =     len(train_data.loader),
            desc =      "Training",
            leave =     False,
            colour =    "cyan"
        ) as epoch_progress:
            
            # # Extract batch from loader.
            # batch =                     next(loader)
            
            # For each batch in loader...
            for batch in train_data.loader:
            
                # Get proper inputs from batch.
                if len(input_keys) > 1:     inputs = {key: batch[key] for key in input_keys}
                else:                       inputs = batch[input_keys[0]]
                
                # Make forward pass through model.
                model_output:   Tensor =    model(inputs)
                
                # Get proper ground truth keys.
                if len(target_keys) > 1:    targets = {key: batch[key] for key in target_keys}
                else:                       targets = batch[target_keys[0]]
                
                # Compute loss.
                loss =  loss_function(model_output, targets)
                
                # Back propagation.
                loss.backward()
                
                # Update weights.
                optimizer.step()
                
                # Reset gradients.
                optimizer.zero_grad()
                
                # Update progress bar.
                epoch_progress.update(1)
        
        # VALIDATE =================================================================================
        
    # TEST =========================================================================================