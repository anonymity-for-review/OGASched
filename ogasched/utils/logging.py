"""
The logging configurations.
"""
import logging


logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[%(asctime)s %(levelname)s] %(lineno)d: %(message)s"
)
