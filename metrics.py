"""Metrics module."""

__all__ = ["calculate_metrics", "plot_results"]

from logging                        import Logger
from os                             import makedirs

import matplotlib.pyplot            as plt

from matplotlib.container           import BarContainer
from matplotlib.pyplot              import bar, figure, subplot, text, tight_layout, savefig, title, xticks
from medpy.metric.binary            import hd
from numpy                          import mean, ndarray, sum, uint8, unique
from pandas                         import DataFrame
from torch                          import Tensor, tensor

from utilities                      import LOGGER, TIMESTAMP

def calculate_metrics(
    prediction:     Tensor,
    target:         Tensor,
    dataset_name:   str,
    num_classes:    int,
    smooth:         float = 1e-6
) -> dict:
    """# Calculate segmentation metrics.
    
    ## Args:
        * prediction    (Tensor):           Model prediction(s).
        * target        (Tensor):           Ground truth(s) to model prediction(s).
        * smooth        (float, optional):  Smoothing factor.
        
    ## Returns:
        * dict: Dice coefficient, precision, recall, and Haussdorf distance.
    """
    # Convert prediction and target to arrays.
    prediction: ndarray =   prediction.detach().cpu().numpy()
    target:     ndarray =   target.detach().cpu().numpy()
    
    # Initialize metrics map.
    metrics:    dict =      {
                                "Dice Score":   [],
                                "Precision":    [],
                                "Recall":       [],
                                "Hausdorff":    []
                            }
    
    # For Pet dataset (1-indexed), we need to adjust our class handling.
    if dataset_name.lower() == "pets":
        
        # Process 1-indexed classes (1, 2, 3)
        class_values:   ndarray =   unique(target)
        min_class:      int =       min(class_values)
        max_class:      int =       max(tensor(class_values))
        class_range:    int =       range(min_class, max_class + 1)
        
    # Otherwise, simply get range.
    else: class_range = range(num_classes)
    
    # For each class in dataset...
    for cls in class_range:
        
        # Convert to integers.
        prediction_cls: ndarray =   (prediction == cls).astype(uint8)
        target_cls:     ndarray =   (target == cls).astype(uint8)
        
        # Skip background for specific datasets 
        if cls == 0 and dataset_name.lower() != "pets":    continue
            
        # Skip if this class is not present in this batch
        if sum(target_cls) == 0:                             continue
        
        # For pet dataset - class 1 is the main class of interest (pet).
        # Weight it higher than background (2) or border (3).
        class_weight:   float =     3.0 if (cls == 1 and dataset_name.lower() == "pets") else 1.0
            
        # Calculate Dice score.
        intersection:   int =       sum(prediction_cls * target_cls)
        dice:           float =     (2. * intersection + smooth) / (sum(prediction_cls) + sum(target_cls) + smooth)
        
        # Append to metrics map.
        metrics["Dice Score"].append(dice * class_weight)
        
        # Calculate true//false-positives & negatives.
        true_positive:  int =       sum(prediction_cls * target_cls)
        false_positive: int =       sum(prediction_cls) - true_positive
        false_negative: int =       sum(target_cls) - true_positive
        
        # Calculate precision & recall.
        precision:      float =     (true_positive + smooth) / (true_positive + false_positive + smooth)
        recall:         float =     (true_positive + smooth) / (true_positive + false_negative + smooth)

        # Append to metrics map.
        metrics["Precision"].append(precision * class_weight)
        metrics["Recall"].append(recall * class_weight)
        
        # If both pred and target have some pixels.
        if sum(prediction_cls) > 0 and sum(target_cls) > 0:
            
            try:# Calculate Haussdorf distance.
                hausdorff:  float = hd(prediction_cls, target_cls)
                
                # Append to metrics map.
                metrics["Hausdorff"].append(hausdorff)
                
                
            # Haussdorff distance calculation can fail if shapes are too complex
            # or if there are no contiguous regions.
            except Exception as e: print(f"Hausdorff calculation failed for class {cls}: {e}")
            
        # Otherise...
        else:
            # Penalize with a high distance if prediction is empty but target is not.
            if sum(target_cls) > 0: metrics["Hausdorff"].append(100.0)  # High penalty
    
    # Initialize averaged results.
    results = {}
    
    # For each metric calculated...
    for metric, values in metrics.items():
        
        # Calculate average if there are values.
        if values:  results[metric] = mean(values)
        
        # Otherwise, simply record a zero.
        else:       results[metric] = 0.0
    
    # Return averaged results.
    return results

def plot_results(
    results_df: DataFrame,
    dataset_name:   str,
    num_classes:    int,
    save_path:  str =       "results"
) -> None:
    """Plot the benchmark results.
    
    ## Args:
        * results_df    (DataFrame):    DataFrame containing benchmark results.
    """
    # Initialize logger.
    _logger_:   Logger =        LOGGER.getChild("plot-results")
    
    # Create a directory for plots.
    makedirs(f"{save_path}/plots", exist_ok = True)
    
    # Initialize figure.
    figure(figsize=(12, 8))
    
    # For each metric recorded...
    for i, metric in enumerate(["Dice Score", "Precision", "Recall", "Hausdorff"]):
        
        # Set subplot location.
        subplot(2, 2, i+1)
        
        # Initialize bar graph.
        bars: BarContainer =    plt.bar(results_df["Model"], results_df[metric])
        
        # Set title.
        title(f"{metric.replace("_", " ").title()}")
        
        # Slant labels.
        xticks(rotation = 45)
        
        # For each bar in graph...
        for bar in bars:
            
            # Record height.
            height: int =   bar.get_height()
            
            # Add values on top of bars.
            text(bar.get_x() + bar.get_width()/2., height, f"{height:.3f}", ha = "center", va = "bottom")
    
    # Initialize new layout.
    tight_layout()
    
    # Save figure.
    savefig(f"{save_path}/{dataset_name}_segmentation_metrics_{TIMESTAMP}.png")
    
    # Initialize figure for computational costs.
    figure(figsize = (10, 5))
    
    # Set subplot location.
    subplot(2, 2, 1)
    
    # Initialize bar graph.
    bars:   BarContainer =  plt.bar(results_df["Model"], results_df["Inference Time MS"])
    
    # Set title.
    title("Inference Time (ms)")
    
    # Slant labels.
    xticks(rotation = 45)
    
    # For each bar in graph...
    for bar in bars:
        
        # Get height of bar.
        height: int =   bar.get_height()
        
        # Add values on top of bars.
        text(bar.get_x() + bar.get_width()/2., height, f"{height:.1f}", ha = "center", va = "bottom")
    
    # Set subplot location.
    subplot(2, 2, 2)
    
    # Initialize bar graph.
    bars:   BarContainer =  plt.bar(results_df["Model"], results_df["Peak Memory MB"])
    
    # Set title.
    title("Peak Memory Usage (MB)")
    
    # Slant labels.
    xticks(rotation = 45)
    
    # For each bar in graph.
    for bar in bars:
        
        # Get height of bar.
        height: int =   bar.get_height()
        
        # Add values on top of bars.
        text(bar.get_x() + bar.get_width()/2., height, f"{height:.1f}", ha = "center", va = "bottom")
    
    # Set subplot location.
    subplot(2, 2, 3)
    
    # Initialize bar graph.
    bars:   BarContainer =  plt.bar(results_df["Model"], results_df["FLOPS"])
    
    # Set title.
    title("Floating Point Operations Per Second")
    
    # Slant labels.
    xticks(rotation = 45)
    
    # For each bar in graph...
    for bar in bars:
        
        # Get height of bar.
        height: int =   bar.get_height()
        
        # Add values on top of bars.
        text(bar.get_x() + bar.get_width()/2., height, f"{height:.1f}", ha = "center", va = "bottom")
    
    # Set subplot location.
    subplot(2, 2, 4)
    
    # Initialize bar graph.
    bars:   BarContainer =  plt.bar(results_df["Model"], results_df["Parameters"])
    
    # Set title.
    title("Parameters")
    
    # Slant labels.
    xticks(rotation = 45)
    
    # For each bar in graph...
    for bar in bars:
        
        # Get height of bar.
        height: int =   bar.get_height()
        
        # Add values on top of bars.
        text(bar.get_x() + bar.get_width()/2., height, f"{height:.1f}", ha = "center", va = "bottom")
    
    # Set new layout.
    tight_layout()
    
    # Save figure.
    savefig(f"{save_path}/{dataset_name}_computational_costs_{TIMESTAMP}.png")
    
    _logger_.info(f"Plots saved to {save_path}/plots/")