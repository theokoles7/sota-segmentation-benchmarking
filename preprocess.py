"""Pre-processing module."""

__all__ = ["preprocess_pets_mask", "precprocess_voc_mask"]

from logging    import Logger

from numpy      import argmax, array, digitize, ndarray, round, uint8
from torch      import from_numpy, Tensor

from utilities  import LOGGER
    
def preprocess_pets_mask(self,
    mask:   Tensor
) -> Tensor:
    """# Process Oxford-IIIT Pet dataset masks.
    
    The pet dataset has 1=pet, 2=background, 3=border.
    We"ll keep the original values to ensure proper segmentation.
    
    ## Args:
        * mask  (Tensor):   Mask being converted.
        
    ## Returns:
        * Tensor:   Converted tensor.
    """
    # If mask is tensor...
    if isinstance(mask, Tensor):
        
        # If all values are close to 0 or 1, this means the normalization happened.
        if max(mask) <= 1.01:
            
            # Convert tensor to numpy for processing.
            mask_np:        ndarray =   mask.squeeze(1).cpu().numpy()
            
            # Get the original mask by multiplying by 255 and rounding.
            # Pet masks should have values 1, 2, 3 (or possibly 0, 1, 2).
            mask_np:        ndarray =   round(mask_np * 255).astype(uint8)
            
            # Check if we have values above 3, if so, it"s probably 0-255 scale.
            if max(mask_np) > 3:
                
                # Scale down values: 1-63 -> 1, 64-128 -> 2, 129-255 -> 3.
                mask_np:    ndarray =   digitize(mask_np, bins = [0, 64, 128, 255])
            
            # Convert back to tensor.
            mask:           Tensor =    from_numpy(mask_np).long()
            
    # Otherwise, convert to tensor.
    else: mask = from_numpy(array(mask)).long()
        
    # Finally, make sure we have the right dimensions.
    if len(mask.shape) == 3 and mask.shape[0] == 1: mask = mask.squeeze(0)
    
    # Return converted mask.
    return mask

def preprocess_voc_mask(self,
    mask:   Tensor
) -> Tensor:
    """# Convert VOC mask to class indices.
    
    ## Args:
        * mask  (Tensor):   Mask being converted.
        
    ## Returns:
        * Tensor:   Converted tensor.
    """
    # Check the input mask shape and format.
    if isinstance(mask, Tensor):
        
        # VOC masks come as RGB tensors - need to convert to class indices
        if len(mask.shape) == 4 and mask.shape[1] == 3:     return argmax(mask, dim=1)
        
        
        elif len(mask.shape) == 3 and mask.shape[0] == 3:   return mask.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
            # Process further if needed
            
        # Ensure mask is 2D for each sample
        if len(mask.shape) == 3 and mask.shape[0] == 1:     return mask.squeeze(0)
            
    else:   return from_numpy(array(mask)).long()
    
    return mask