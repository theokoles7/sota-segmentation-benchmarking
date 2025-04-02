"""Argument definitions for U-Net model."""

__all__ = ["add_unet_parser"]

from argparse   import ArgumentParser, _SubParsersAction

def add_unet_parser(
    parent_subparser:   _SubParsersAction
) -> None:
    """# Add parser/arguments for U-Net model.

    ## Args:
        * parent_subparser  (_SubParsersAction):    Parent's sub-parser.
    """
    # Initialize parser.
    _parser_:   ArgumentParser =    parent_subparser.add_parser(
                                        name =  "unet",
                                        help =  """U-Net (CNN-based) segmentation model."""
                                    )
    
    # Add arguments.
    _parser_.add_argument(
        "--channels-out", "-co",
        type =      int,
        nargs =     "+",
        default =   [1, 256, 256],
        help =      """Output channels of the model. Defaults to [1, 256, 256]."""
    )
    
    _parser_.add_argument(
        "--encoder-channels", "-ec",
        type =      int,
        nargs =     "+",
        default =   [3, 16, 32, 64],
        help =      """Encoder channels of the model. Defaults to [3, 16, 32, 64]."""
    )
    
    _parser_.add_argument(
        "--decoder-channels", "-dc",
        type =      int,
        nargs =     "+",
        default =   [64, 32, 16],
        help =      """Decoder channels of the model. Defaults to [64, 32, 16]."""
    )
    
    _parser_.add_argument(
        "--segmentation-classes", "-sc",
        type =      int,
        default =   1,
        help =      """Segmentation classes of the model. Defaults to 1."""
    )
    
    _parser_.add_argument(
        "--retain-dimension", "-rd",
        action =    "store_true",
        default =   True,
        help =      """Retain dimension of the model. Defaults to True."""
    )