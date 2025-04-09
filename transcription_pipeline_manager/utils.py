import sys
import argparse
import logging
import requests
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential
from .constants import (
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF,
    DOWNLOAD_TIMEOUT,
)

def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


def fail_hard(message: str) -> None:
    logger = logging.getLogger(__name__)
    logger.error(message)
    sys.exit(1)


@retry(
    stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=DEFAULT_RETRY_BACKOFF),
)
def post_request(url: str, data: dict[str, Any], json: bool = False) -> requests.Response:
    kwargs: dict[str, Any] = {
        "timeout": DOWNLOAD_TIMEOUT,
    }
    if json:
        kwargs["json"] = data
    else:
        kwargs["data"] = data
    response = requests.post(url, **kwargs)
    response.raise_for_status()
    return response
