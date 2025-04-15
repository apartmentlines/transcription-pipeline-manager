import sys
import argparse
import logging
import requests
from typing import Any
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed
from .constants import (
    POST_RETRY_ATTEMPTS,
    POST_RETRY_TIMEOUT,
    POST_RETRY_WAIT_FIXED,
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
    stop=(stop_after_attempt(POST_RETRY_ATTEMPTS) | stop_after_delay(POST_RETRY_TIMEOUT)),
    wait=wait_fixed(POST_RETRY_WAIT_FIXED),
)
def post_request(url: str, data: dict[str, Any], json: bool = False) -> requests.Response:
    """
    Sends a POST request with configurable retry logic.

    :param url: The URL to send the POST request to.
    :type url: str
    :param data: The data payload for the request.
    :type data: dict[str, Any]
    :param json: If True, send data as JSON payload; otherwise, send as form data. Defaults to False.
    :type json: bool
    :return: The response object from the successful request.
    :rtype: requests.Response
    :raises: tenacity.RetryError: If the request fails after all retry attempts.
             requests.exceptions.RequestException: For other request-related errors.
    """
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
