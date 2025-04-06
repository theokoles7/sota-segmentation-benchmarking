"""SAM model wrapper.

Implementation adapted from https://pypi.org/project/segment-anything-py/
"""

from logging            import Logger

from segment_anything   import SamPredictor

from utilities          import LOGGER

class SAM(SamPredictor):
    """SAM wrapper class."""
    
    def __init__(self,

    ):
        
        SamPredictor()