"""Benchmarking process."""

__all__ = ["run_benchmark"]

from json                           import dumps
from logging                        import Logger
from time                           import time

from numpy                          import isnan, round
from pandas                         import DataFrame
from thop                           import profile
from torch                          import argmax, int64, max, no_grad, randn, Tensor
from torch.amp                      import autocast, GradScaler
from torch.cuda                     import empty_cache, is_available, max_memory_allocated, memory_allocated
from torchvision.datasets           import OxfordIIITPet, VOCSegmentation
from torch.nn                       import Module
from torch.nn.functional            import softmax
from torch.utils.data               import DataLoader
from tqdm                           import tqdm

from datasets                       import load_dataset
from metrics                        import calculate_metrics, plot_results
from models                         import load_model
from preprocess                     import preprocess_pets_mask, preprocess_voc_mask
from utilities                      import LOGGER, TIMESTAMP

def run_benchmark(
    dataset_name:   str =           "pets",
    num_classes:    int =           3,
    models:         list[str] =     ["deeplab-v3", "fpn", "seg-former", "u-net"],
    batch_size:     int =           8, 
    num_workers:    int =           4,
    device:         str =           "cuda",
    save_path:      str =           "results",
    use_amp:        bool =          True,
    input_size:     tuple[int] =    (512, 512),
    **kwargs
) -> DataFrame:
    """# Run the benchmark on all models and compile results.

    ## Args:
        * dataset_name  (str, optional):        Dataset on which model(s) will be evaluated. 
                                                Options are "pets", "coco", and "voc". Defaults 
                                                to "pets".
        * models        (list[str], optional):  List of models being evaluated. Options are 
                                                "unet", "deeplabv3", "fpn". Defaults to None.
        * batch_size    (int, optional):        Dataloader batch size. Defaults to 8.
        * num_workers   (int, optional):        Number of threads to use for data loading. 
                                                Defaults to 4.
        * device        (str, optional):        One of "cuda" or "cpu". Defaults to "cuda".
        * save_path     (str, optional):        Path at which evaluation report(s) will be 
                                                saved. Defaults to "results".
        * use_amp       (bool, optional):       Use autocast mixed precision. Defaults to True.
    
    ## Returns:
        * DataFrame:    Metrics report.
    """
    # Initialize logger.
    _logger_:   Logger =                            LOGGER.getChild("benchmark")
    
    # Initialize results array.
    results:    list =                              []
    
    # Load dataset.
    dataloader: DataLoader =                        load_dataset(
                                                        dataset_name =  dataset_name,
                                                        batch_size =    batch_size,
                                                        num_workers =   num_workers,
                                                        input_size =    input_size
                                                    )
    
    # Log dataset info for debugging.
    _logger_.debug(f"""Dataset:        {dataset_name}
    Number of samples:  {len(dataloader)}
    Number of classes:  {num_classes}
    Batch size:         {batch_size}
    Device:             {device}
    Mixed Precision:    {use_amp}""")
    
    # For each model in the list...
    for model_name in models:
        
        # Log action.
        _logger_.info(f"Evaluating {model_name} on {dataset_name}.")
        
        # Initialize model.
        model:                  Module =        load_model(model_name)
        
        # Set model to evaluation mode.
        model.eval()
        
        # Calulcate dimensions of input.
        input:                  Tensor =        randn(
                                                    batch_size, 
                                                    3, 
                                                    input_size[0], 
                                                    input_size[1]
                                                ).to(device)
        
        # Calculate FLOPs
        flops, parameters =                     profile(
                                                    model =         model,
                                                    inputs =        (input,),
                                                    verbose =       False
                                                )
        
        _logger_.info(f"FLOPS: {flops}, PARAMETERS: {parameters}")
        
        # Initialize metrics map.
        metrics_sum:            dict =          {
                                                    "Dice Score":   0.0,
                                                    "Precision":    0.0,
                                                    "Recall":       0.0,
                                                    "Hausdorff":    0.0
                                                }
        
        # Initialize count of iterations.
        metrics_count:          int =           0
        
        # Track computational costs.
        total_inference_time:   float =         0
        start_memory:           int =           memory_allocated(device) if is_available() else 0
        
        # Initialize gradient scaler.
        scaler:                 GradScaler =    GradScaler()
        
        # Without calculating gradients...
        with no_grad():
            
            # For each sample...
            for i, data in enumerate(tqdm(iterable = dataloader, desc = f"{model_name} on {dataset_name}", colour = "magenta")):
                
                # If working on VOC dataset...
                if dataset_name.lower() == "voc":
                    
                    # Extract image and mask.
                    images, masks = data
                    
                    # Preprocess mask.
                    masks:  Tensor =    preprocess_voc_mask(masks)
                    
                # Otherwise, for pets dataset...
                elif dataset_name.lower() == "pets":
                    
                    # Extract image and mask.
                    images, masks = data
                    
                    # Don"t preprocess if our target_transform already gives proper masks.
                    if max(masks) <= 1.0: masks = preprocess_pets_mask(masks)
                    
                # Set image & mask to device.
                images:     Tensor =    images.to(device)
                masks:      Tensor =    masks.to(device)
                
                # Record starting time to record inference.
                start_time: float =     time()
                
                # If using mixed precision...
                if use_amp:
                    
                    # Forward pass with autocast.
                    with autocast("cuda"):  outputs = model(images)
                    
                # Regular forward pass otherwise.
                else: outputs = model(images)
                
                # Record final inference time.
                inference_time: float =     time() - start_time
                
                # Accumulate.
                total_inference_time +=     inference_time
                
                # Apply softmax to get probabilities
                outputs:        Tensor =    softmax(outputs, dim=1)
                
                # Get predictions
                preds:          Tensor =    argmax(outputs, dim=1)
                
                # Adjust predictions to match target numbering for Pet dataset.
                if dataset_name.lower() == "pets": preds = preds + 1
                
                # Calculate metrics.
                batch_metrics:  dict =      calculate_metrics(
                                                prediction =    preds,
                                                target =        masks,
                                                dataset_name =  dataset_name,
                                                num_classes =   num_classes
                                            )
                
                # Update metrics sum.
                for metric, value in batch_metrics.items():
                    if not isnan(value): metrics_sum[metric] += value
                
                # Accumulate iterations count.
                metrics_count += 1
        
        # Calculate average metrics.
        avg_metrics:        dict =  {k: v / metrics_count for k, v in metrics_sum.items()}
        
        # Calculate memory usage.
        peak_memory:        int =   (max_memory_allocated(device) - start_memory) / (1024 * 1024)
            
        # Calculate average inference time.
        avg_inference_time: float = total_inference_time / metrics_count
        
        # Record results
        model_results:      dict =  {
                                        "Model":                model_name,
                                        "Dice Score":           avg_metrics["Dice Score"],
                                        "Precision":            avg_metrics["Precision"],
                                        "Recall":               avg_metrics["Recall"],
                                        "Hausdorff":            avg_metrics["Hausdorff"],
                                        "Inference Time MS":    avg_inference_time * 1000,
                                        "Peak Memory MB":       peak_memory,
                                        "FLOPS":                flops,
                                        "Parameters":           parameters
                                    }
        
        # Append results to report.
        results.append(model_results)
        
        # Log final results.
        _logger_.info(f"Results for {model_name}: {dumps(model_results, indent = 2, default = str)}")
        
        # Clear GPU memory.
        if is_available(): empty_cache()
    
    # Create a pandas DataFrame for easy analysis.
    results_df: DataFrame = DataFrame(results)
    
    # Save report to CSV.
    results_df.to_csv(
        path_or_buf =   f"{save_path}/{dataset_name}_benchmark_results_{TIMESTAMP}.csv",
        index =         False
    )
    
    # Generate plots.
    plot_results(
        results_df =    results_df,
        dataset_name =  dataset_name,
        num_classes =   num_classes
    )
    
    # Return results.
    return results_df