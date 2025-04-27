"""Argument definitions for Pascal VOC Segmentation dataset."""

__all__ = ["add_voc_parser"]

from argparse   import ArgumentParser, _SubParsersAction

def add_voc_parser(
    parent_subparser:   _SubParsersAction
) -> None:
    """# Add parser/arguments for Pascal VOC Segmentation scan dataset.

    ## Args:
        * parent_subparser  (_SubParsersAction): Parent's sub-parser.
    """
    # Initialize parser.
    _parser_:   ArgumentParser =    parent_subparser.add_parser(
                                        name =  "voc",
                                        help =  """Pascal VOC Segmentation dataset."""
                                    )