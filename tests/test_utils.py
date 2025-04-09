import argparse
import sys
import pytest
import requests
import tenacity
from unittest.mock import patch, Mock
from transcription_pipeline_manager.utils import (
    positive_int,
    fail_hard,
    post_request,
)

from transcription_pipeline_manager.constants import (
    DEFAULT_RETRY_ATTEMPTS,
    DOWNLOAD_TIMEOUT,
)


def test_positive_int():
    assert positive_int("5") == 5
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")


def test_fail_hard(caplog):
    with patch.object(sys, "exit") as mock_exit:
        fail_hard("Test error")
        mock_exit.assert_called_with(1)
        assert "Test error" in caplog.text


@patch("requests.post")
def test_post_request_success(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response
    url = "http://example.com"
    data = {"key": "value"}
    response = post_request(url, data=data)
    assert response.status_code == 200
    mock_post.assert_called_once_with(url, data=data, timeout=DOWNLOAD_TIMEOUT)


@patch("requests.post")
@patch("tenacity.nap.time.sleep")
def test_post_request_failure(_, mock_post):
    mock_post.side_effect = requests.exceptions.RequestException
    url = "http://example.com"
    data = {"key": "value"}
    with pytest.raises(tenacity.RetryError):
        post_request(url, data=data)
    mock_post.assert_called_with(url, data=data, timeout=DOWNLOAD_TIMEOUT)
    assert mock_post.call_count == DEFAULT_RETRY_ATTEMPTS


@patch("requests.post")
def test_post_request_json_success(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response
    url = "http://example.com"
    data = {"key": "value"}
    response = post_request(url, data=data, json=True)
    assert response.status_code == 200
    mock_post.assert_called_once_with(url, json=data, timeout=DOWNLOAD_TIMEOUT)


@patch("requests.post")
@patch("tenacity.nap.time.sleep")
def test_post_request_json_failure(_, mock_post):
    mock_post.side_effect = requests.exceptions.RequestException
    url = "http://example.com"
    data = {"key": "value"}
    with pytest.raises(tenacity.RetryError):
        post_request(url, data=data, json=True)
    mock_post.assert_called_with(url, json=data, timeout=DOWNLOAD_TIMEOUT)
    assert mock_post.call_count == DEFAULT_RETRY_ATTEMPTS
