"""Argument definitions for Oxford IIIT PETS dataset."""

__all__ = ["add_pets_parser"]

from argparse   import ArgumentParser, _SubParsersAction

def add_pets_parser(
    parent_subparser:   _SubParsersAction
) -> None:
    """# Add parser/arguments for Oxford IIIT PETS scan dataset.

    ## Args:
        * parent_subparser  (_SubParsersAction): Parent's sub-parser.
    """
    # Initialize parser.
    _parser_:   ArgumentParser =    parent_subparser.add_parser(
                                        name =  "pets",
                                        help =  """Oxford IIIT PETS dataset."""
                                    )