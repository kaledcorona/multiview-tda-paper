"""
Module Description:
-------------------
This module provides utilities for configuring and logging the experiments.
These utilities ensure that logs are appropriately written 
to a file (with timestamped filenames) and optionally to the console.

Functions:
- `setup_logging`: Configures logging for the application.
- `start_experiment_logging`: Initiate loggin for the experiment.
- `log_experiment_metadata`: Output experiment metadata to logs.
- `setup_file_handler`: Creates and configures a file handler for logging.
- `configure_console_handler`: Sets up a console handler for logging to stdout.

This module is essential for managing logging in larger applications, especially
when debugging or tracking the execution flow of programs with different log levels.

Note:
    For using it in the experiment, only call start_experiment_logging:
        >>> logger = start_experiment_logging(Path('/path/to/dir'))

Author: kaled Corona
Date: 2025-02-17
"""


# ============================
# Standard Library Imports
# ============================
import logging
from pathlib import Path
from datetime import datetime

# ============================
# Third-Party Library Imports
# ============================


# ============================
# Local Application Imports
# ============================
from experiment.files import ensure_dir_exists, generate_filename

# ============================
# Configuration & Settings
# ============================


# ============================
# Main Execution 
# ============================


def setup_logging(log_dir: Path = Path('logs'),
                  log_level = logging.INFO, 
                  log_to_console: bool = True) -> logging.Logger:
    """
    Sets up the logging configuration, including file and console handlers.

    This function configures the root logger to write log messages to a file and, optionally, to the console.
    It ensures the specified log directory exists, generates a timestamped log filename, and attaches the 
    necessary handlers to the logger.

    Args:
        log_dir (Path, optional): The directory where the log file will be saved. Defaults to 'logs'.
        log_level (int, optional): The logging level to be used (e.g., logging.INFO, logging.DEBUG).
                                   Defaults to logging.INFO.
        log_to_console (bool, optional): Whether to also log messages to the console. Defaults to True.

    Returns:
        logging.Logger: The configured logger object, which can be used to log messages.

    Raises:
        FileNotFoundError: If the specified directory cannot be created or accessed.
        ValueError: If the generated log filename is invalid or the logging configuration fails.

    Example:
        >>> logger = setup_logging(log_dir=Path("/tmp/logs"), log_level=logging.DEBUG)
        >>> logger.info("This is an informational message.")
        >>> logger.debug("This is a debug message.")
    
    Note:
        - The log filename is automatically generated using a timestamp.
        - The logger will write logs to the file and, if `log_to_console` is True, also to the console.
        - The function assumes the `generate_filename`, `setup_file_handler`, and `configure_console_handler`
          functions are defined elsewhere in the program.
    """
    ensure_dir_exists(log_dir)
    

    filename: Path = generate_filename(log_dir, name="experiment_logs", format="log")

    logger = logging.getLogger(str(filename))
    
    logger.setLevel(log_level)

    # Ensure no duplicate handlers are added to this logger
    if not logger.hasHandlers():
        handler = setup_file_handler(filename, log_level)
        logger.addHandler(handler)

    if log_to_console:
        console_handler = configure_console_handler(log_level)
        logger.addHandler(console_handler)

    logger.info(f"Logging initialized.\n Writing to {filename}")
    return logger


def setup_file_handler(file: Path, log_level: int) -> logging.FileHandler:
    """
    Sets up a file handler for logging, writing log messages to a specified file.

    This function creates a logging `FileHandler` that writes log messages to the
    specified file and formats the log messages with timestamps and log levels.
    The logging level for the handler is set according to the provided `log_level`.

    Args:
        file (Path): The path to the log file where the log messages will be written.
                      If the file does not exist, it will be created.
        log_level (int): The logging level to set for the file handler. This corresponds
                         to one of the levels in the `logging` module (e.g., `logging.INFO`,
                         `logging.DEBUG`, `logging.WARNING`).

    Returns:
        logging.FileHandler: A `FileHandler` object configured with the specified log level
                              and the formatter for logging.

    Example:
        >>> log_file = Path("/path/to/logfile.log")
        >>> handler = setup_file_handler(log_file, logging.DEBUG)
        >>> logger = logging.getLogger()
        >>> logger.addHandler(handler)
        >>> logger.debug("This is a debug message.")
    
    Note:
        The log file will be created if it doesn't exist. Ensure the provided path is
        valid and accessible to avoid potential errors.
    """
    handler = logging.FileHandler(file)
    handler.setLevel(log_level)
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    return handler


def configure_console_handler(log_level: int) -> logging.StreamHandler:
    """Create and configure a console handler."""
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", 
                                  datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    return console_handler


def start_experiment_logging(log_dir: Path = Path('logs')) -> logging.Logger:
    """
    Starts the logging environment for the experiment and log initial message.

    It is a wrapper around setup_logging to start it and log an initial message
    for the experiment.

    Args:
        log_dir (Path): represents a directory path using pathlib's Path.

    Returns:
        logging.Logger: an logger instance.

    Examples:
        Basic usage example:
            >>> logger = start_experiment_logging(Path('/path/to/dir'))
            >>> logger.info("Some info")
    """
    logger = setup_logging(log_dir, log_to_console=True)
    logger.info("========== [Experiment Start] ==========")
    return logger


def log_experiment_metadata(logger: logging.Logger, 
                            NUMBER_EXPERIMENTS: int,
                            CHUNK_SIZE: int,
                            RANDOM_STATE: int) -> None:
    """
    Logs the metadata of the experiment, including the number of experiments,
    chunk size, and random state.

    Args:
        logger (logging.Logger): The logger object to log the information.
        NUMBER_EXPERIMENTS (int): The number of experiments to run.
        CHUNK_SIZE (int): The size of each data chunk.
        RANDOM_STATE (int): The random state for reproducibility.

    Returns:
        None: This function does not return a value; it only logs the metadata.
    """
    logger.info(
    f"Experiment configurations:\n"
    f"  NUMBER_EXPERIMENTS: {NUMBER_EXPERIMENTS}\n"
    f"  CHUNK_SIZE: {CHUNK_SIZE}\n"
    f"  RANDOM_STATE: {RANDOM_STATE}\n"
    )