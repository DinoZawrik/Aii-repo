"""
Logging configuration module.

This module sets up structured logging using structlog for better observability.
"""

import logging
import sys
from typing import Optional
import structlog
from structlog.typing import EventDict, WrappedLogger


def add_log_level(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Add log level to the event dict.

    Args:
        logger: The wrapped logger instance.
        method_name: Name of the logging method.
        event_dict: Event dictionary.

    Returns:
        Modified event dictionary with log level.
    """
    if method_name == "warn":
        method_name = "warning"
    event_dict["level"] = method_name.upper()
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "console",
    service_name: str = "sentiment-api"
) -> None:
    """
    Setup structured logging with structlog.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Format of logs - 'console' for human-readable, 'json' for JSON.
        service_name: Name of the service for logging context.
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Choose processors based on format
    if log_format == "json":
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:  # console format
        processors = [
            structlog.contextvars.merge_contextvars,
            add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Add service name to context
    structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a logger instance.

    Args:
        name: Name of the logger (typically __name__ of the module).

    Returns:
        Configured structlog logger.
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()
