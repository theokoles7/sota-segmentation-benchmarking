"""Argument definitions & parsing."""

__all__ = ["ARGS"]

from argparse                   import ArgumentParser, _ArgumentGroup, Namespace, _SubParsersAction

# Initialize parser
_parser_:       ArgumentParser =    ArgumentParser(
    prog =          "segment-benchmark",
    description =   """CSCE-508 (Image Processing) project focues on benchmarking the 
                    state-of-the-art segmentation models (as of 2025)."""
)

# Initialize sub-parser
_subparser_:    _SubParsersAction = _parser_.add_subparsers(
    dest =          "cmd",
    help =          """Command being executed."""
)

# +================================================================================================+
# | BEGIN ARGUMENTS                                                                                |
# +================================================================================================+

# LOGGING ==========================================================================================
_logging_:      _ArgumentGroup =    _parser_.add_argument_group(
    title =         "Logging",
    description =   """Logging configuration."""
)

_logging_.add_argument(
    "--logging-level",
    type =          str,
    choices =       ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default =       "INFO",
    help =          """Minimum logging level (DEBUG < INFO < WARNING < ERROR < CRITICAL). Defaults 
                    to "INFO"."""
)

_logging_.add_argument(
    "--logging-path",
    type =          str,
    default =       "logs",
    help =          """Path at which log files will be written. Defaults to "./logs/"."""
)

# OUTPUT ===========================================================================================
_output_:       _ArgumentGroup =    _parser_.add_argument_group(
    title =         "Output",
    description =   """Output/reporting configuration."""
)

_output_.add_argument(
    "--output-path",
    type =          str,
    default =       "output",
    help =          """Path at which output/report files will be written. Defaults to "./output/"."""
)

# COMMANDS =========================================================================================

# +================================================================================================+
# | END ARGUMENTS                                                                                  |
# +================================================================================================+

# Parse arguments
ARGS:           Namespace =         _parser_.parse_args()