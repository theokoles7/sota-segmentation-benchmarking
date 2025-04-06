"""Argument definitions for running a job."""

__all__ = ["add_run_job_parser"]

from argparse                   import ArgumentParser, _SubParsersAction

from utilities.arguments.models import *

def add_run_job_parser(
    parent_subparser:   _SubParsersAction
) -> None:
    """# Add parser/arguments for running a job.

    ## Args:
        * parent_subparser  (_SubParsersAction):    Parent's sub-parser.
    """
    # Initialize parser.
    _parser_:   ArgumentParser =        parent_subparser.add_parser(
                                            name =  "run-job",
                                            help =  """Run a job on the CellMap Challenge dataset."""
                                        )
    
    _parser_.add_argument(
        "datasplit",
        type =      int,
        choices =   [1, 2, 3, 4, 5],
        default =   1,
        help =      """Datasplit to run the job on. Defaults to 1."""
    )
    
    _parser_.add_argument(
        "--epochs",
        type =      int,
        default =   200,
        help =      """Number of epochs for which training phase will execute."""
    )
    
    # Initialize sub-parser.
    _subparser_:  _SubParsersAction =   _parser_.add_subparsers(
                                            dest =          "model",
                                            description =   """Model with which job will be executed."""
                                        )
    
    # Add model parsers.
    add_unet_parser(parent_subparser = _subparser_)