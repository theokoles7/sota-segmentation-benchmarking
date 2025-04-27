"""Dataset arguments package."""

__all__ = ["add_pets_parser", "add_voc_parser"]

from utilities.arguments.datasets.pets  import add_pets_parser
from utilities.arguments.datasets.voc   import add_voc_parser