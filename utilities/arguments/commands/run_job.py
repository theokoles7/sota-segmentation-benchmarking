"""Argument definitions for running a job."""

__all__ = ["add_run_job_parser"]

from argparse   import ArgumentParser, _SubParsersAction

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
    
    # Initialize sub-parser.
    _subparser_:  _SubParsersAction =   _parser_.add_subparsers(
                                            title =     "model",
                                            description = """Model with which job will be executed."""
                                        )