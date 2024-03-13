import logging

log = logging.getLogger(__name__)


def my_func(i):
    log.info("Running my_func")
    log.debug(f"Processing on iteration: {i+1}")
