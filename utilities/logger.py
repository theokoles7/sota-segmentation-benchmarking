"""Logging utilities."""

__all__ = ["LOGGER"]

from logging                import getLogger, Formatter, Logger, StreamHandler
from logging.handlers       import RotatingFileHandler
from os                     import makedirs
from sys                    import stdout

from utilities.arguments    import ARGS
from utilities.timestamp    import TIMESTAMP

# Ensure that logging path exists
makedirs(name = ARGS.logging_path, exist_ok = True)

# Initialize logger
LOGGER:         Logger =                getLogger("segment")

# Set logging level
LOGGER.setLevel(ARGS.logging_level)

# Define console handler
stdout_handler: StreamHandler =         StreamHandler(stdout)
stdout_handler.setFormatter(Formatter("%(levelname)s | %(name)s | %(message)s"))
LOGGER.addHandler(stdout_handler)

# Define file handler
file_handler:   RotatingFileHandler =   RotatingFileHandler(f"{ARGS.logging_path}/default_{TIMESTAMP}.log", maxBytes = 1048576)
file_handler.setFormatter(Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
LOGGER.addHandler(file_handler)