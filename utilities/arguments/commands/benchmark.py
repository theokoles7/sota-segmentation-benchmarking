"""Argument definitions for running a job."""

__all__ = ["add_run_job_parser"]

from argparse                       import ArgumentParser, _SubParsersAction

from utilities.arguments.datasets   import *

def add_benchmark_parser(
    parent_subparser:   _SubParsersAction
) -> None:
    """# Add parser/arguments for running a job.

    ## Args:
        * parent_subparser  (_SubParsersAction):    Parent's sub-parser.
    """
    # Initialize parser.
    _parser_:   ArgumentParser =        parent_subparser.add_parser(
                                            name =  "benchmark",
                                            help =  """Benchmark segmentation model(s) on a dataset."""
                                        )
    
    # Initialize sub-parser.
    _subparser_:  _SubParsersAction =   _parser_.add_subparsers(
                                            dest =          "model",
                                            description =   """Model with which job will be executed."""
                                        )
    
    # Add dataset parsers.
    add_pets_parser(parent_subparser =  _subparser_)
    add_voc_parser( parent_subparser =  _subparser_)