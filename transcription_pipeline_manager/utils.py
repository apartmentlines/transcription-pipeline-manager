import sys
import argparse
import logging

def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


def fail_hard(message: str) -> None:
    logger = logging.getLogger(__name__)
    logger.error(message)
    sys.exit(1)
