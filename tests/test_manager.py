import argparse
import logging
import time
import json
import threading
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import Any

import pytest
from pytest_mock import MockerFixture
import requests
from requests.exceptions import RequestException
from tenacity import RetryError

# Modules to test
from transcription_pipeline_manager import constants as const
from transcription_pipeline_manager import manager as manager_module
from transcription_pipeline_manager.manager import TranscriptionPipelineManager
from transcription_pipeline_manager.rest_interface import RestInterface
from transcription_pipeline_manager.logger import Logger

# --- Test Constants ---
TEST_API_KEY = "test-api-key-123"
TEST_DOMAIN = "test.example.com"
TEST_POD_ID = "test-pod-id-456"
TEST_POD_IP = "192.168.1.100"
TEST_POD_PORT = 8080
TEST_POD_URL = f"http://{TEST_POD_IP}:{TEST_POD_PORT}"


# --- Fixtures ---

@pytest.fixture
def mock_logger(mocker: MockerFixture) -> MagicMock:
    """Mocks the Logger class."""
    mock = mocker.patch("transcription_pipeline_manager.manager.Logger", autospec=True)
    mock_instance = MagicMock(spec=logging.Logger)
    mock.return_value = mock_instance
    return mock_instance

@pytest.fixture
def mock_rest_interface(mocker: MockerFixture) -> MagicMock:
    """Mocks the RestInterface class."""
    mock = mocker.patch("transcription_pipeline_manager.manager.RestInterface", autospec=True)
    mock_instance = MagicMock(spec=RestInterface)
    mock.return_value = mock_instance
    mock_instance.update_pods_total = MagicMock()
    mock_instance.update_pods_running = MagicMock()
    mock_instance.update_pipeline_last_run_time = MagicMock()
    mock_instance.start = MagicMock()
    mock_instance.shutdown = MagicMock()
    return mock_instance

@pytest.fixture
def mock_runpod_manager(mocker: MockerFixture) -> tuple[MagicMock, MagicMock]:
    """Mocks the RunpodSingletonManager class, returning start and terminate instances."""
    mock = mocker.patch("transcription_pipeline_manager.manager.RunpodSingletonManager", autospec=True)

    mock_start_instance = MagicMock()
    mock_terminate_instance = MagicMock()

    # Configure the side_effect to return the correct instance based on args
    def side_effect(*args, **kwargs):
        if kwargs.get("stop") is True or kwargs.get("terminate") is True:
            return mock_terminate_instance
        else:
            return mock_start_instance

    mock.side_effect = side_effect

    mock_start_instance.run = MagicMock(return_value=TEST_POD_ID)
    mock_start_instance.count_pods = MagicMock(return_value={'total': 1, 'running': 1})
    mock_terminate_instance.run = MagicMock(return_value=True)

    return mock_start_instance, mock_terminate_instance


@pytest.fixture
def manager_instance(mocker: MockerFixture) -> TranscriptionPipelineManager:
    """
    Creates a TranscriptionPipelineManager instance for testing helper methods.
    Mocks dependencies *after* instantiation, as __init__ is tested separately.
    """
    mock_logger_instance = MagicMock(spec=logging.Logger)
    mock_rest_instance = MagicMock(spec=RestInterface)
    mock_start_manager_instance = MagicMock()
    mock_terminate_manager_instance = MagicMock()

    real_path = Path
    mocker.patch("transcription_pipeline_manager.manager.Path", real_path)

    manager = TranscriptionPipelineManager(
        api_key=TEST_API_KEY,
        domain=TEST_DOMAIN,
        limit=10,
        processing_limit=2,
        ngrok=False,
        debug=False,
    )

    manager.log = mock_logger_instance
    manager.rest_interface = mock_rest_instance
    manager.runpod_start_manager = mock_start_manager_instance
    manager.runpod_terminate_manager = mock_terminate_manager_instance

    mock_rest_instance.update_pods_total = MagicMock()
    mock_rest_instance.update_pods_running = MagicMock()
    mock_rest_instance.update_pipeline_last_run_time = MagicMock()
    mock_rest_instance.start = MagicMock()
    mock_rest_instance.shutdown = MagicMock()
    mock_start_manager_instance.run = MagicMock(return_value=TEST_POD_ID)
    mock_start_manager_instance.count_pods = MagicMock(return_value={'total': 1, 'running': 1})
    mock_terminate_manager_instance.run = MagicMock(return_value=True)

    return manager

# Fixtures specifically for providing the *mocked instances* to helper tests
# These are needed because the manager_instance fixture now creates mocks internally

@pytest.fixture
def mock_logger(manager_instance: TranscriptionPipelineManager) -> MagicMock:
    """Returns the mocked logger instance from the manager_instance fixture."""
    return manager_instance.log

@pytest.fixture
def mock_rest_interface(manager_instance: TranscriptionPipelineManager) -> MagicMock:
    """Returns the mocked RestInterface instance from the manager_instance fixture."""
    return manager_instance.rest_interface

@pytest.fixture
def mock_runpod_manager(manager_instance: TranscriptionPipelineManager) -> tuple[MagicMock, MagicMock]:
    """Returns the mocked RunpodSingletonManager instances from the manager_instance fixture."""
    return manager_instance.runpod_start_manager, manager_instance.runpod_terminate_manager


# --- Test Cases ---

# Phase 2: Initialization (__init__)

@patch("transcription_pipeline_manager.manager.Path")
@patch("transcription_pipeline_manager.manager.RunpodSingletonManager")
@patch("transcription_pipeline_manager.manager.RestInterface")
@patch("transcription_pipeline_manager.manager.Logger")
def test_manager_init_defaults(
    mock_logger_cls: MagicMock,
    mock_rest_interface_cls: MagicMock,
    mock_runpod_manager_cls: MagicMock,
    mock_path_cls: MagicMock,
    mocker: MockerFixture
) -> None:
    """Test TranscriptionPipelineManager initialization with default values."""
    mock_logger_instance = MagicMock(spec=logging.Logger)
    mock_logger_cls.return_value = mock_logger_instance
    mock_rest_instance = MagicMock(spec=RestInterface)
    mock_rest_interface_cls.return_value = mock_rest_instance
    mock_start_manager_instance = MagicMock()
    mock_terminate_manager_instance = MagicMock()
    mock_runpod_manager_cls.side_effect = [mock_start_manager_instance, mock_terminate_manager_instance]

    mock_path_instance = MagicMock(spec=Path)
    mock_path_instance.parent.parent = mock_path_instance
    mock_path_instance.__truediv__.return_value = mock_path_instance
    mock_path_cls.return_value = mock_path_instance

    manager = TranscriptionPipelineManager(
        api_key=TEST_API_KEY,
        domain=TEST_DOMAIN,
    )

    assert manager.api_key == TEST_API_KEY
    assert manager.domain == TEST_DOMAIN
    assert manager.limit == const.DEFAULT_TRANSCRIPTION_LIMIT
    assert manager.processing_limit == const.DEFAULT_TRANSCRIPTION_PROCESSING_LIMIT
    assert manager.ngrok is False
    assert manager.debug is False

    assert manager.callback_url_subdomain == f"www.{TEST_DOMAIN}"
    assert manager.transcription_subdomain == f"my.{TEST_DOMAIN}"
    assert manager.logs_callback_url == f"https://www.{TEST_DOMAIN}/api/transcription/logs?api_key={TEST_API_KEY}"

    mock_logger_cls.assert_called_once_with("TranscriptionPipelineManager", debug=False)
    assert manager.log is mock_logger_instance

    mock_path_cls.assert_called_once_with(manager_module.__file__)
    expected_config_path = mock_path_instance / const.RUNPOD_CONFIG_DIR / const.RUNPOD_CONFIG_FILENAME
    assert manager.runpod_config_path == expected_config_path

    mock_rest_interface_cls.assert_called_once_with(
        host=const.DEFAULT_REST_HOST,
        port=const.DEFAULT_REST_PORT,
        api_key=TEST_API_KEY,
        debug=False
    )
    assert manager.rest_interface is mock_rest_instance

    expected_runpod_calls = [
        call(
            config_path=expected_config_path,
            debug=False,
        ),
        call(
            config_path=expected_config_path,
            debug=False,
            terminate=True
        ),
    ]
    mock_runpod_manager_cls.assert_has_calls(expected_runpod_calls)
    assert mock_runpod_manager_cls.call_count == 2
    assert manager.runpod_start_manager is mock_start_manager_instance
    assert manager.runpod_terminate_manager is mock_terminate_manager_instance

    assert isinstance(manager.shutdown_event, threading.Event)
    assert not manager.shutdown_event.is_set()


@patch("transcription_pipeline_manager.manager.Path")
@patch("transcription_pipeline_manager.manager.RunpodSingletonManager")
@patch("transcription_pipeline_manager.manager.RestInterface")
@patch("transcription_pipeline_manager.manager.Logger")
def test_manager_init_custom_values(
    mock_logger_cls: MagicMock,
    mock_rest_interface_cls: MagicMock,
    mock_runpod_manager_cls: MagicMock,
    mock_path_cls: MagicMock,
    mocker: MockerFixture
) -> None:
    """Test TranscriptionPipelineManager initialization with custom values."""
    mock_logger_instance = MagicMock(spec=logging.Logger)
    mock_logger_cls.return_value = mock_logger_instance
    mock_rest_instance = MagicMock(spec=RestInterface)
    mock_rest_interface_cls.return_value = mock_rest_instance
    mock_start_manager_instance = MagicMock()
    mock_terminate_manager_instance = MagicMock()
    mock_runpod_manager_cls.side_effect = [mock_start_manager_instance, mock_terminate_manager_instance]
    mock_path_instance = MagicMock(spec=Path)
    mock_path_instance.parent.parent = mock_path_instance
    mock_path_instance.__truediv__.return_value = mock_path_instance
    mock_path_cls.return_value = mock_path_instance

    custom_limit = 50
    custom_processing_limit = 4
    custom_ngrok = True
    custom_debug = True

    manager = TranscriptionPipelineManager(
        api_key=TEST_API_KEY,
        domain=TEST_DOMAIN,
        limit=custom_limit,
        processing_limit=custom_processing_limit,
        ngrok=custom_ngrok,
        debug=custom_debug,
    )

    assert manager.api_key == TEST_API_KEY
    assert manager.domain == TEST_DOMAIN
    assert manager.limit == custom_limit
    assert manager.processing_limit == custom_processing_limit
    assert manager.ngrok is custom_ngrok
    assert manager.debug is custom_debug

    mock_logger_cls.assert_called_once_with("TranscriptionPipelineManager", debug=custom_debug)

    mock_rest_interface_cls.assert_called_once_with(
        host=const.DEFAULT_REST_HOST,
        port=const.DEFAULT_REST_PORT,
        api_key=TEST_API_KEY,
        debug=custom_debug
    )

    expected_config_path = mock_path_instance / const.RUNPOD_CONFIG_DIR / const.RUNPOD_CONFIG_FILENAME
    expected_runpod_calls = [
        call(
            config_path=expected_config_path,
            debug=custom_debug,
        ),
        call(
            config_path=expected_config_path,
            debug=custom_debug,
            terminate=True
        ),
    ]
    mock_runpod_manager_cls.assert_has_calls(expected_runpod_calls)
    assert mock_runpod_manager_cls.call_count == 2


# Phase 1: Helper Methods

# _check_pod_idle_status
@patch("transcription_pipeline_manager.manager.requests.get")
def test_check_pod_idle_status_success(mock_get: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _check_pod_idle_status returns True when pod is idle."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "idle"}
    mock_get.return_value = mock_response

    result = manager_instance._check_pod_idle_status(TEST_POD_URL)

    assert result is True
    mock_get.assert_called_once_with(
        f"{TEST_POD_URL}/status",
        timeout=const.POD_REQUEST_TIMEOUT
    )


@patch("transcription_pipeline_manager.manager.requests.get")
def test_check_pod_idle_status_not_idle(mock_get: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _check_pod_idle_status returns False when pod status is not 'idle'."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "loading"}
    mock_get.return_value = mock_response

    result = manager_instance._check_pod_idle_status(TEST_POD_URL)

    assert result is False
    mock_get.assert_called_once_with(
        f"{TEST_POD_URL}/status",
        timeout=const.POD_REQUEST_TIMEOUT
    )


@patch("transcription_pipeline_manager.manager.requests.get")
def test_check_pod_idle_status_http_error(mock_get: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _check_pod_idle_status returns False on non-200 status code."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 404
    mock_response.reason = "Not Found"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mock_get.return_value = mock_response

    result = manager_instance._check_pod_idle_status(TEST_POD_URL)

    assert result is False
    mock_get.assert_called_once_with(
        f"{TEST_POD_URL}/status",
        timeout=const.POD_REQUEST_TIMEOUT
    )
    expected_msg = f"Failed to get pod status from {TEST_POD_URL}/status. Status: 404 Not Found"
    assert mock_logger.debug.call_args_list.count(call(expected_msg)) == 1


@patch("transcription_pipeline_manager.manager.requests.get")
def test_check_pod_idle_status_json_decode_error(mock_get: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _check_pod_idle_status returns False on JSONDecodeError."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)
    mock_get.return_value = mock_response

    result = manager_instance._check_pod_idle_status(TEST_POD_URL)

    assert result is False
    mock_get.assert_called_once_with(
        f"{TEST_POD_URL}/status",
        timeout=const.POD_REQUEST_TIMEOUT
    )
    mock_logger.error.assert_called_once_with(
        f"Failed to decode JSON response from {TEST_POD_URL}/status.", exc_info=False
    )


@patch("transcription_pipeline_manager.manager.requests.get")
def test_check_pod_idle_status_request_exception(mock_get: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _check_pod_idle_status returns False on RequestException."""
    mock_get.side_effect = RequestException("Connection error")

    result = manager_instance._check_pod_idle_status(TEST_POD_URL)

    assert result is False
    mock_get.assert_called_once_with(
        f"{TEST_POD_URL}/status",
        timeout=const.POD_REQUEST_TIMEOUT
    )
    mock_logger.error.assert_called_once_with(
        f"Error requesting pod status from {TEST_POD_URL}/status: Connection error", exc_info=False
    )


# _trigger_pipeline_run
@patch("transcription_pipeline_manager.manager.post_request")
@patch("transcription_pipeline_manager.manager.time.time", return_value=1234567890)
def test_trigger_pipeline_run_success(mock_time: MagicMock, mock_post_request: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _trigger_pipeline_run returns True on successful API call."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True, "message": "Pipeline run triggered successfully"}
    mock_post_request.return_value = mock_response

    expected_payload = {
        "api_key": TEST_API_KEY,
        "domain": manager_instance.transcription_subdomain,
        "limit": manager_instance.limit,
        "processing_limit": manager_instance.processing_limit,
        "callback_url": manager_instance.logs_callback_url,
    }

    result = manager_instance._trigger_pipeline_run(TEST_POD_URL)

    assert result is True
    mock_post_request.assert_called_once_with(
        f"{TEST_POD_URL}/run",
        expected_payload,
        True
    )
    mock_rest_interface.update_pipeline_last_run_time.assert_called_once_with(1234567890)
    mock_logger.info.assert_any_call(f"Pipeline run triggered successfully at {TEST_POD_URL}/run: Pipeline run triggered successfully")


@patch("transcription_pipeline_manager.manager.post_request")
def test_trigger_pipeline_run_failure_response(mock_post_request: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _trigger_pipeline_run returns False when API response indicates failure."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": False, "message": "Invalid parameters"}
    mock_post_request.return_value = mock_response

    result = manager_instance._trigger_pipeline_run(TEST_POD_URL)

    assert result is False
    mock_post_request.assert_called_once()
    mock_logger.error.assert_called_once_with(
        f"Pipeline run trigger failed at {TEST_POD_URL}/run: Invalid parameters"
    )
    mock_rest_interface.update_pipeline_last_run_time.assert_not_called()


@patch("transcription_pipeline_manager.manager.post_request")
def test_trigger_pipeline_run_json_decode_error(mock_post_request: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _trigger_pipeline_run returns False on JSONDecodeError."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    json_exception = json.JSONDecodeError("Expecting value", "doc", 0)
    mock_response.json.side_effect = json_exception
    mock_post_request.return_value = mock_response

    result = manager_instance._trigger_pipeline_run(TEST_POD_URL)

    assert result is False
    mock_post_request.assert_called_once()
    mock_logger.error.assert_called_once_with(
        f"Failed to decode JSON response from {TEST_POD_URL}/run: {json_exception}", exc_info=False
    )
    mock_rest_interface.update_pipeline_last_run_time.assert_not_called()


@patch("transcription_pipeline_manager.manager.post_request")
def test_trigger_pipeline_run_unexpected_json(mock_post_request: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _trigger_pipeline_run returns False on unexpected JSON structure (missing 'success')."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    json_return_value = {"message": "Something happened, but no success key"}
    mock_response.json.return_value = json_return_value
    mock_post_request.return_value = mock_response

    result = manager_instance._trigger_pipeline_run(TEST_POD_URL)

    assert result is False
    mock_post_request.assert_called_once()
    mock_logger.error.assert_called_once_with(
        f"Unexpected JSON response format from {TEST_POD_URL}/run (missing 'success' key): {json_return_value}"
    )
    mock_rest_interface.update_pipeline_last_run_time.assert_not_called()


@patch("transcription_pipeline_manager.manager.post_request")
def test_trigger_pipeline_run_invalid_success_type(mock_post_request: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _trigger_pipeline_run returns False when 'success' key is not a boolean."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    json_return_value = {"success": "not_a_boolean", "message": "Invalid success type"}
    mock_response.json.return_value = json_return_value
    mock_post_request.return_value = mock_response

    result = manager_instance._trigger_pipeline_run(TEST_POD_URL)

    assert result is False
    mock_post_request.assert_called_once()
    mock_logger.error.assert_called_once_with(
        f"Invalid type for 'success' key in response from {TEST_POD_URL}/run (expected bool, got str): {json_return_value}"
    )
    mock_rest_interface.update_pipeline_last_run_time.assert_not_called()


# Phase 3: State Handler Unit Tests (_handle_...)

def test_handle_starting_cycle(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _handle_starting_cycle returns correct initial state values."""
    now = time.time()
    result = manager_instance._handle_starting_cycle(now)
    assert len(result) == 7
    next_state, cycle_start_time, pod_id, pod_url, last_count, last_idle, wait_start = result

    assert next_state == const.STATE_ATTEMPTING_POD_START
    assert cycle_start_time == now
    assert pod_id == ""
    assert pod_url == ""
    assert last_count == 0.0
    assert last_idle == 0.0
    assert wait_start == 0.0
    mock_logger.debug.assert_called_with("Starting new hourly cycle.")


def test_handle_attempting_pod_start_success(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_runpod_manager: tuple[MagicMock, MagicMock]) -> None:
    """Test _handle_attempting_pod_start success path."""
    mock_start_manager, _ = mock_runpod_manager
    mock_start_manager.run.return_value = TEST_POD_ID
    expected_url = const.POD_URL_TEMPLATE % TEST_POD_ID

    with patch.object(manager_instance, '_terminate_pods') as mock_terminate:
        next_state, pod_id, pod_url = manager_instance._handle_attempting_pod_start()

    assert next_state == const.STATE_WAITING_FOR_IDLE
    assert pod_id == TEST_POD_ID
    assert pod_url == expected_url
    mock_start_manager.run.assert_called_once()
    mock_terminate.assert_not_called()


def test_handle_attempting_pod_start_failure_no_pod_info(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_runpod_manager: tuple[MagicMock, MagicMock]) -> None:
    """Test _handle_attempting_pod_start failure when runpod manager returns None."""
    mock_start_manager, _ = mock_runpod_manager
    mock_start_manager.run.return_value = None

    with patch.object(manager_instance, '_terminate_pods') as mock_terminate:
        next_state, pod_id, pod_url = manager_instance._handle_attempting_pod_start()

    assert next_state == const.STATE_WAITING_AFTER_FAILURE
    assert pod_id == ""
    assert pod_url == ""
    mock_start_manager.run.assert_called_once()
    mock_terminate.assert_called_once()


def test_handle_attempting_pod_start_failure_exception(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_runpod_manager: tuple[MagicMock, MagicMock]) -> None:
    """Test _handle_attempting_pod_start failure on exception."""
    mock_start_manager, _ = mock_runpod_manager
    test_exception = Exception("API Error")
    mock_start_manager.run.side_effect = test_exception

    with patch.object(manager_instance, '_terminate_pods') as mock_terminate:
        next_state, pod_id, pod_url = manager_instance._handle_attempting_pod_start()

    assert next_state == const.STATE_WAITING_AFTER_FAILURE
    assert pod_id == ""
    assert pod_url == ""
    mock_start_manager.run.assert_called_once()
    mock_terminate.assert_called_once()


def test_handle_waiting_for_idle_timeout(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _handle_waiting_for_idle timeout condition."""
    now = time.time()
    wait_for_idle_start_time = now - const.STATUS_CHECK_TIMEOUT - 1
    last_idle_check_time = now - 10

    with patch.object(manager_instance, '_terminate_pods') as mock_terminate:
        next_state, updated_last_idle = manager_instance._handle_waiting_for_idle(
            now, wait_for_idle_start_time, last_idle_check_time, TEST_POD_ID, TEST_POD_URL
        )

    assert next_state == const.STATE_WAITING_AFTER_FAILURE
    assert updated_last_idle == last_idle_check_time # Time should not update on timeout
    mock_terminate.assert_called_once()
    expected_elapsed = now - wait_for_idle_start_time
    expected_log_msg = f"Timeout ({const.STATUS_CHECK_TIMEOUT}s, waited {expected_elapsed:.1f}s) waiting for pod {TEST_POD_ID} to become idle. Terminating."
    mock_logger.warning.assert_called_once_with(expected_log_msg)


def test_handle_waiting_for_idle_interval_not_reached(manager_instance: TranscriptionPipelineManager) -> None:
    """Test _handle_waiting_for_idle when check interval is not reached."""
    now = time.time()
    wait_for_idle_start_time = now - 100
    last_idle_check_time = now - (const.POD_STATUS_CHECK_INTERVAL / 2) # Interval not reached

    with patch.object(manager_instance, '_check_pod_idle_status') as mock_check_idle:
        with patch.object(manager_instance, '_terminate_pods') as mock_terminate:
            next_state, updated_last_idle = manager_instance._handle_waiting_for_idle(
                now, wait_for_idle_start_time, last_idle_check_time, TEST_POD_ID, TEST_POD_URL
            )

    assert next_state == const.STATE_WAITING_FOR_IDLE
    assert updated_last_idle == last_idle_check_time # Time should not update
    mock_check_idle.assert_not_called()
    mock_terminate.assert_not_called()


def test_handle_waiting_for_idle_check_not_idle(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _handle_waiting_for_idle when pod check returns not idle."""
    now = time.time()
    wait_for_idle_start_time = now - 100
    last_idle_check_time = now - (const.POD_STATUS_CHECK_INTERVAL * 2) # Interval reached

    with patch.object(manager_instance, '_check_pod_idle_status', return_value=False) as mock_check_idle:
        with patch.object(manager_instance, '_terminate_pods') as mock_terminate:
            next_state, updated_last_idle = manager_instance._handle_waiting_for_idle(
                now, wait_for_idle_start_time, last_idle_check_time, TEST_POD_ID, TEST_POD_URL
            )

    assert next_state == const.STATE_WAITING_FOR_IDLE
    assert updated_last_idle == now # Time should update
    mock_check_idle.assert_called_once_with(TEST_POD_URL)
    mock_terminate.assert_not_called()


def test_handle_waiting_for_idle_check_is_idle(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _handle_waiting_for_idle when pod check returns idle."""
    now = time.time()
    wait_for_idle_start_time = now - 100
    last_idle_check_time = now - (const.POD_STATUS_CHECK_INTERVAL * 2) # Interval reached

    with patch.object(manager_instance, '_check_pod_idle_status', return_value=True) as mock_check_idle:
        with patch.object(manager_instance, '_terminate_pods') as mock_terminate:
            next_state, updated_last_idle = manager_instance._handle_waiting_for_idle(
                now, wait_for_idle_start_time, last_idle_check_time, TEST_POD_ID, TEST_POD_URL
            )

    assert next_state == const.STATE_ATTEMPTING_PIPELINE_RUN
    assert updated_last_idle == now # Time should update
    mock_check_idle.assert_called_once_with(TEST_POD_URL)
    mock_terminate.assert_not_called()


def test_handle_attempting_pipeline_run_success(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _handle_attempting_pipeline_run success path."""
    with patch.object(manager_instance, '_trigger_pipeline_run', return_value=True) as mock_trigger:
        with patch.object(manager_instance, '_terminate_pods') as mock_terminate:
            next_state = manager_instance._handle_attempting_pipeline_run(TEST_POD_ID, TEST_POD_URL)

    assert next_state == const.STATE_UPDATING_COUNTS
    mock_trigger.assert_called_once_with(TEST_POD_URL)
    mock_terminate.assert_not_called()


def test_handle_attempting_pipeline_run_failure(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _handle_attempting_pipeline_run failure path."""
    with patch.object(manager_instance, '_trigger_pipeline_run', return_value=False) as mock_trigger:
        with patch.object(manager_instance, '_terminate_pods') as mock_terminate:
            next_state = manager_instance._handle_attempting_pipeline_run(TEST_POD_ID, TEST_POD_URL)

    assert next_state == const.STATE_WAITING_AFTER_FAILURE
    mock_trigger.assert_called_once_with(TEST_POD_URL)
    mock_terminate.assert_called_once()


def test_handle_updating_counts_interval_not_reached(manager_instance: TranscriptionPipelineManager) -> None:
    """Test _handle_updating_counts when interval is not reached."""
    now = time.time()
    last_count_update_time = now - (const.COUNT_UPDATE_INTERVAL / 2) # Interval not reached

    with patch.object(manager_instance, '_update_pod_counts') as mock_update:
        next_state, updated_last_count = manager_instance._handle_updating_counts(now, last_count_update_time)

    assert next_state == const.STATE_UPDATING_COUNTS
    assert updated_last_count == last_count_update_time # Time should not update
    mock_update.assert_not_called()


def test_handle_updating_counts_interval_reached_update_success(manager_instance: TranscriptionPipelineManager) -> None:
    """Test _handle_updating_counts when interval reached and update succeeds."""
    now = time.time()
    last_count_update_time = now - (const.COUNT_UPDATE_INTERVAL * 2) # Interval reached

    with patch.object(manager_instance, '_update_pod_counts', return_value=True) as mock_update:
        next_state, updated_last_count = manager_instance._handle_updating_counts(now, last_count_update_time)

    assert next_state == const.STATE_UPDATING_COUNTS
    assert updated_last_count == now # Time should update
    mock_update.assert_called_once()


def test_handle_updating_counts_interval_reached_update_failure(manager_instance: TranscriptionPipelineManager) -> None:
    """Test _handle_updating_counts when interval reached and update fails."""
    now = time.time()
    last_count_update_time = now - (const.COUNT_UPDATE_INTERVAL * 2) # Interval reached

    with patch.object(manager_instance, '_update_pod_counts', return_value=False) as mock_update:
        next_state, updated_last_count = manager_instance._handle_updating_counts(now, last_count_update_time)

    assert next_state == const.STATE_UPDATING_COUNTS
    assert updated_last_count == last_count_update_time # Time should not update
    mock_update.assert_called_once()


def test_handle_waiting_after_failure(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _handle_waiting_after_failure returns correct state and logs."""
    elapsed_cycle_time = 1000
    remaining_time = max(0, const.CYCLE_DURATION - elapsed_cycle_time)

    next_state = manager_instance._handle_waiting_after_failure(elapsed_cycle_time)

    assert next_state == const.STATE_WAITING_AFTER_FAILURE
    mock_logger.warning.assert_called_with(f"In failure state. Waiting for next cycle. Time remaining: {remaining_time:.0f}s")


# Phase 3: run() Method Integration Tests

# NOTE: These tests now call run() directly, mocking instance methods
# 'get_current_time' and 'sleep' to control the loop execution.

@patch.object(TranscriptionPipelineManager, "sleep", return_value=None)
@patch.object(TranscriptionPipelineManager, "get_current_time")
@patch.object(TranscriptionPipelineManager, "_setup_signal_handlers")
@patch.object(TranscriptionPipelineManager, "_shutdown")
def test_run_initialization_and_immediate_shutdown(
    mock_setup_signals: MagicMock,
    mock_shutdown: MagicMock,
    mock_get_current_time: MagicMock,
    mock_sleep: MagicMock,
    manager_instance: TranscriptionPipelineManager,
    mock_rest_interface: MagicMock, # Fixture provides the mock instance
    mocker: MockerFixture,
) -> None:
    """Test run() calls setup methods and REST start, then shuts down."""
    # Stop the loop immediately
    mocker.patch.object(manager_instance.shutdown_event, 'is_set', return_value=True)
    # Provide a time value for the single loop check
    mock_get_current_time.return_value = 1000.0

    manager_instance.run() # Call directly

    mock_setup_signals.assert_called_once()
    mock_rest_interface.start.assert_called_once()
    # _shutdown is called in finally block
    mock_shutdown.assert_called_once()


# Phase 4: Argument Parsing (parse_arguments)

def test_parse_arguments_defaults(mocker: MockerFixture) -> None:
    """Test parse_arguments with default values."""
    mocker.patch("sys.argv", ["script_name"]) # Simulate no command-line args
    args = manager_module.parse_arguments()
    assert args.api_key is None
    assert args.domain is None
    assert args.limit == const.DEFAULT_TRANSCRIPTION_LIMIT
    assert args.processing_limit == const.DEFAULT_TRANSCRIPTION_PROCESSING_LIMIT
    assert args.ngrok is False
    assert args.debug is False

def test_parse_arguments_custom_values(mocker: MockerFixture) -> None:
    """Test parse_arguments with custom values provided."""
    test_key = "custom_key"
    test_domain = "custom.domain"
    test_limit = 500
    test_processing_limit = 5
    mocker.patch("sys.argv", [
        "script_name",
        "--api-key", test_key,
        "--domain", test_domain,
        "--limit", str(test_limit),
        "--processing-limit", str(test_processing_limit),
        "--ngrok",
        "--debug",
    ])
    args = manager_module.parse_arguments()
    assert args.api_key == test_key
    assert args.domain == test_domain
    assert args.limit == test_limit
    assert args.processing_limit == test_processing_limit
    assert args.ngrok is True
    assert args.debug is True

@pytest.mark.parametrize("arg_name", ["--limit", "--processing-limit"])
def test_parse_arguments_invalid_positive_int(mocker: MockerFixture, arg_name: str) -> None:
    """Test parse_arguments raises error for non-positive integer values."""
    mocker.patch("sys.argv", ["script_name", arg_name, "0"])
    with pytest.raises(SystemExit):
        manager_module.parse_arguments()

    mocker.patch("sys.argv", ["script_name", arg_name, "-10"])
    with pytest.raises(SystemExit):
        manager_module.parse_arguments()

    mocker.patch("sys.argv", ["script_name", arg_name, "not_an_int"])
    with pytest.raises(SystemExit):
        manager_module.parse_arguments()


# Phase 5: Entry Point (main)

@patch("transcription_pipeline_manager.manager.parse_arguments")
@patch("transcription_pipeline_manager.manager.load_configuration")
@patch("transcription_pipeline_manager.manager.TranscriptionPipelineManager")
@patch("transcription_pipeline_manager.manager.fail_hard")
def test_main_success_flow(
    mock_fail_hard: MagicMock,
    mock_manager_cls: MagicMock,
    mock_load_config: MagicMock,
    mock_parse_args: MagicMock,
) -> None:
    """Test the main function executes the success path correctly."""
    mock_args = argparse.Namespace(
        api_key="cli_key", domain="cli.domain", limit=10, processing_limit=2, ngrok=False, debug=False
    )
    mock_parse_args.return_value = mock_args
    mock_load_config.return_value = (TEST_API_KEY, TEST_DOMAIN)
    mock_pipeline_instance = MagicMock(spec=TranscriptionPipelineManager)
    mock_manager_cls.return_value = mock_pipeline_instance

    manager_module.main()

    mock_parse_args.assert_called_once()
    mock_load_config.assert_called_once_with(mock_args)
    mock_manager_cls.assert_called_once_with(
        api_key=TEST_API_KEY,
        domain=TEST_DOMAIN,
        limit=mock_args.limit,
        processing_limit=mock_args.processing_limit,
        ngrok=mock_args.ngrok,
        debug=mock_args.debug,
    )
    mock_pipeline_instance.run.assert_called_once()
    mock_fail_hard.assert_not_called()


@patch("transcription_pipeline_manager.manager.parse_arguments")
@patch("transcription_pipeline_manager.manager.load_configuration")
@patch("transcription_pipeline_manager.manager.TranscriptionPipelineManager")
@patch("transcription_pipeline_manager.manager.fail_hard")
@patch("transcription_pipeline_manager.manager.logging.getLogger")
def test_main_load_config_failure(
    mock_get_logger: MagicMock,
    mock_fail_hard: MagicMock,
    mock_manager_cls: MagicMock,
    mock_load_config: MagicMock,
    mock_parse_args: MagicMock,
) -> None:
    """Test main function calls fail_hard if load_configuration fails."""
    mock_args = argparse.Namespace(debug=False) # Need debug flag for logger check
    mock_parse_args.return_value = mock_args
    test_exception = ValueError("Missing config")
    mock_load_config.side_effect = test_exception
    mock_logger_instance = MagicMock()
    mock_get_logger.return_value = mock_logger_instance

    manager_module.main()

    mock_parse_args.assert_called_once()
    mock_load_config.assert_called_once_with(mock_args)
    mock_manager_cls.assert_not_called()
    mock_logger_instance.critical.assert_called_once_with(
        f"Unhandled exception during manager execution: {test_exception}", exc_info=False
    )
    mock_fail_hard.assert_called_once_with("Transcription Pipeline Manager failed due to an unhandled exception.")


@patch("transcription_pipeline_manager.manager.parse_arguments")
@patch("transcription_pipeline_manager.manager.load_configuration")
@patch("transcription_pipeline_manager.manager.TranscriptionPipelineManager")
@patch("transcription_pipeline_manager.manager.fail_hard")
@patch("transcription_pipeline_manager.manager.logging.getLogger")
def test_main_manager_init_failure(
    mock_get_logger: MagicMock,
    mock_fail_hard: MagicMock,
    mock_manager_cls: MagicMock,
    mock_load_config: MagicMock,
    mock_parse_args: MagicMock,
) -> None:
    """Test main function calls fail_hard if manager initialization fails."""
    mock_args = argparse.Namespace(
        api_key="k", domain="d", limit=1, processing_limit=1, ngrok=False, debug=True
    )
    mock_parse_args.return_value = mock_args
    mock_load_config.return_value = (TEST_API_KEY, TEST_DOMAIN)
    test_exception = TypeError("Bad init arg")
    mock_manager_cls.side_effect = test_exception
    mock_logger_instance = MagicMock()
    mock_get_logger.return_value = mock_logger_instance

    manager_module.main()

    mock_parse_args.assert_called_once()
    mock_load_config.assert_called_once()
    mock_manager_cls.assert_called_once()
    mock_logger_instance.critical.assert_called_once_with(
         f"Unhandled exception during manager execution: {test_exception}", exc_info=True
    )
    mock_fail_hard.assert_called_once_with("Transcription Pipeline Manager failed due to an unhandled exception.")


@patch("transcription_pipeline_manager.manager.parse_arguments")
@patch("transcription_pipeline_manager.manager.load_configuration")
@patch("transcription_pipeline_manager.manager.TranscriptionPipelineManager")
@patch("transcription_pipeline_manager.manager.fail_hard")
@patch("transcription_pipeline_manager.manager.logging.getLogger")
def test_main_manager_run_failure(
    mock_get_logger: MagicMock,
    mock_fail_hard: MagicMock,
    mock_manager_cls: MagicMock,
    mock_load_config: MagicMock,
    mock_parse_args: MagicMock,
) -> None:
    """Test main function calls fail_hard if pipeline.run() fails."""
    mock_args = argparse.Namespace(
        api_key="k", domain="d", limit=1, processing_limit=1, ngrok=False, debug=False
    )
    mock_parse_args.return_value = mock_args
    mock_load_config.return_value = (TEST_API_KEY, TEST_DOMAIN)
    mock_pipeline_instance = MagicMock(spec=TranscriptionPipelineManager)
    test_exception = ConnectionError("Network down")
    mock_pipeline_instance.run.side_effect = test_exception
    mock_manager_cls.return_value = mock_pipeline_instance
    mock_logger_instance = MagicMock()
    mock_get_logger.return_value = mock_logger_instance

    manager_module.main()

    mock_parse_args.assert_called_once()
    mock_load_config.assert_called_once()
    mock_manager_cls.assert_called_once()
    mock_pipeline_instance.run.assert_called_once()
    mock_logger_instance.critical.assert_called_once_with(
         f"Unhandled exception during manager execution: {test_exception}", exc_info=False
    )
    mock_fail_hard.assert_called_once_with("Transcription Pipeline Manager failed due to an unhandled exception.")


@patch.object(TranscriptionPipelineManager, "sleep", return_value=None)
@patch.object(TranscriptionPipelineManager, "get_current_time")
@patch.object(TranscriptionPipelineManager, "_handle_starting_cycle")
@patch.object(TranscriptionPipelineManager, "_handle_attempting_pod_start")
@patch.object(TranscriptionPipelineManager, "_handle_waiting_for_idle")
@patch.object(TranscriptionPipelineManager, "_handle_attempting_pipeline_run")
@patch.object(TranscriptionPipelineManager, "_handle_updating_counts")
@patch.object(TranscriptionPipelineManager, "_shutdown")
@patch.object(TranscriptionPipelineManager, "_setup_signal_handlers")
def test_run_navigates_full_success_cycle(
    mock_setup_signals: MagicMock,
    mock_shutdown: MagicMock,
    mock_handle_updating: MagicMock,
    mock_handle_attempt_run: MagicMock,
    mock_handle_wait_idle: MagicMock,
    mock_handle_attempt_start: MagicMock,
    mock_handle_start_cycle: MagicMock,
    mock_get_current_time: MagicMock,
    mock_sleep: MagicMock,
    manager_instance: TranscriptionPipelineManager,
    mocker: MockerFixture,
) -> None:
    """Test run() successfully navigates the main state sequence directly."""
    # Let the loop run enough times for all states, then stop
    num_loops = 7 # Start -> Attempt -> Wait(x2) -> Run -> Update -> Update(stay)
    mocker.patch.object(manager_instance.shutdown_event, 'is_set', side_effect=[False] * num_loops + [True])

    # Simulate time advancing slightly each loop
    mock_get_current_time.side_effect = [1000.0 + i for i in range(num_loops + 1)]
    start_time = 1000.0

    # --- Mock State Transitions ---
    mock_handle_start_cycle.return_value = (const.STATE_ATTEMPTING_POD_START, start_time, "", "", 0.0, 0.0, 0.0)
    mock_handle_attempt_start.return_value = (const.STATE_WAITING_FOR_IDLE, "pod", "url")
    # Simulate one check returning not idle, then idle
    mock_handle_wait_idle.side_effect = [
        (const.STATE_WAITING_FOR_IDLE, start_time + 1), # Loop 3: returns wait
        (const.STATE_ATTEMPTING_PIPELINE_RUN, start_time + 2), # Loop 4: returns run
    ]
    mock_handle_attempt_run.return_value = const.STATE_UPDATING_COUNTS # Loop 5
    # Stay in updating counts
    mock_handle_updating.return_value = (const.STATE_UPDATING_COUNTS, start_time + 3) # Loop 6, 7
    # --- End Mock ---

    manager_instance.run() # Call directly

    # Assert handlers were called
    mock_handle_start_cycle.assert_called_once()
    mock_handle_attempt_start.assert_called_once()
    assert mock_handle_wait_idle.call_count == 2
    mock_handle_attempt_run.assert_called_once()
    assert mock_handle_updating.call_count >= 1 # Called in loops 6 and 7

    mock_shutdown.assert_called_once()


@patch.object(TranscriptionPipelineManager, "sleep", return_value=None)
@patch.object(TranscriptionPipelineManager, "get_current_time")
@patch.object(TranscriptionPipelineManager, "_handle_starting_cycle")
@patch.object(TranscriptionPipelineManager, "_handle_attempting_pod_start")
@patch.object(TranscriptionPipelineManager, "_handle_waiting_for_idle")
@patch.object(TranscriptionPipelineManager, "_shutdown")
@patch.object(TranscriptionPipelineManager, "_setup_signal_handlers")
def test_run_triggers_hourly_reset(
    mock_setup_signals: MagicMock,
    mock_shutdown: MagicMock,
    mock_handle_wait_idle: MagicMock,
    mock_handle_attempt_start: MagicMock,
    mock_handle_start_cycle: MagicMock,
    mock_get_current_time: MagicMock,
    mock_sleep: MagicMock,
    manager_instance: TranscriptionPipelineManager,
    mocker: MockerFixture,
) -> None:
    """Test run() resets to STARTING_CYCLE after CYCLE_DURATION."""
    # Loop sequence:
    # 1: Start -> Attempt
    # 2: Attempt -> Wait
    # 3: Wait -> Wait (Time < CYCLE_DURATION)
    # 4: Wait -> Wait (Time >= CYCLE_DURATION -> Reset happens BEFORE handler)
    # 5: Start -> Attempt (Loop stops)
    num_loops = 5
    mocker.patch.object(manager_instance.shutdown_event, 'is_set', side_effect=[False] * num_loops + [True])

    # Simulate time: Start, small increments, then jump past CYCLE_DURATION
    start_time = 1000.0
    time_before_reset = start_time + const.CYCLE_DURATION - 10
    time_after_reset = start_time + const.CYCLE_DURATION + 10
    mock_get_current_time.side_effect = [
        start_time,     # Loop 1 'now'
        start_time + 1, # Loop 2 'now'
        time_before_reset, # Loop 3 'now' (before timeout)
        time_after_reset,  # Loop 4 'now' (after timeout, triggers reset check)
        time_after_reset + 1, # Loop 5 'now'
        time_after_reset + 2, # Extra time for shutdown check
    ]

    # --- Mock State Transitions ---
    # Configure start cycle to be callable multiple times
    def start_cycle_side_effect(now):
        # The 'now' passed here is the mocked time for the current loop iteration
        return (const.STATE_ATTEMPTING_POD_START, now, "", "", 0.0, 0.0, 0.0)
    mock_handle_start_cycle.side_effect = start_cycle_side_effect

    mock_handle_attempt_start.return_value = (const.STATE_WAITING_FOR_IDLE, "pod", "url")

    # Stay in WAITING_FOR_IDLE - use side_effect to get current time
    def wait_idle_side_effect(now, elapsed, last_idle, *args):
        # This will be called in loop 3 (time < reset) and loop 4 (time >= reset)
        # The reset check in run() happens *before* this handler in loop 4
        return (const.STATE_WAITING_FOR_IDLE, now) # Return current time as last_check_time
    mock_handle_wait_idle.side_effect = wait_idle_side_effect
    # --- End Mock ---

    manager_instance.run() # Call directly

    # Assertions
    assert mock_handle_start_cycle.call_count == 2 # Called in loop 1 and loop 5 (after reset)
    mock_handle_attempt_start.assert_called() # Called in loop 1 and 5
    assert mock_handle_wait_idle.call_count >= 1 # Called in loop 3
    mock_shutdown.assert_called_once()


@patch.object(TranscriptionPipelineManager, "sleep", return_value=None)
@patch.object(TranscriptionPipelineManager, "get_current_time")
@patch.object(TranscriptionPipelineManager, "_handle_starting_cycle")
@patch.object(TranscriptionPipelineManager, "_handle_attempting_pod_start")
@patch.object(TranscriptionPipelineManager, "_handle_waiting_for_idle")
@patch.object(TranscriptionPipelineManager, "_shutdown")
@patch.object(TranscriptionPipelineManager, "_setup_signal_handlers")
def test_run_shuts_down_cleanly_on_event(
    mock_setup_signals: MagicMock,
    mock_shutdown: MagicMock,
    mock_handle_wait_idle: MagicMock,
    mock_handle_attempt_start: MagicMock,
    mock_handle_start_cycle: MagicMock,
    mock_get_current_time: MagicMock,
    mock_sleep: MagicMock,
    manager_instance: TranscriptionPipelineManager,
    mocker: MockerFixture,
) -> None:
    """Test run() shuts down cleanly when the event is set."""
    # Loop sequence:
    # 1: Start -> Attempt
    # 2: Attempt -> Wait
    # 3: Wait -> Wait (Event is set after this loop)
    # 4: Loop terminates on event check
    num_loops = 3
    # Set the event after num_loops iterations
    shutdown_side_effect = [False] * num_loops
    def set_event_then_check(*args):
        if not shutdown_side_effect: # If list is empty, event should be set
            return True
        result = shutdown_side_effect.pop(0)
        if not shutdown_side_effect: # If that was the last False, set the real event
             manager_instance.shutdown_event.set()
        return result
    mock_event_is_set = mocker.patch.object(manager_instance.shutdown_event, 'is_set', side_effect=set_event_then_check)

    mock_get_current_time.side_effect = [1000.0 + i for i in range(num_loops + 1)]

    # --- Mock State Transitions ---
    start_time = 1000.0
    mock_handle_start_cycle.return_value = (const.STATE_ATTEMPTING_POD_START, start_time, "", "", 0.0, 0.0, 0.0)
    mock_handle_attempt_start.return_value = (const.STATE_WAITING_FOR_IDLE, "pod", "url")
    mock_handle_wait_idle.return_value = (const.STATE_WAITING_FOR_IDLE, start_time + 1.0)
    # --- End Mock ---

    manager_instance.run() # Call directly

    # Assertions
    assert mock_handle_wait_idle.call_count >= 1 # Should have run at least once
    assert mock_event_is_set.call_count == num_loops + 1 # Called until it returns True
    assert manager_instance.shutdown_event.is_set() # Verify the real event was set
    mock_shutdown.assert_called_once()


@patch.object(TranscriptionPipelineManager, "sleep", return_value=None)
@patch.object(TranscriptionPipelineManager, "get_current_time")
@patch.object(TranscriptionPipelineManager, "_handle_starting_cycle")
@patch.object(TranscriptionPipelineManager, "_handle_attempting_pod_start")
@patch.object(TranscriptionPipelineManager, "_check_pod_idle_status") # Mock the check it calls
@patch.object(TranscriptionPipelineManager, "_handle_waiting_after_failure")
@patch.object(TranscriptionPipelineManager, "_terminate_pods") # Need to mock this as it's called by the handler
@patch.object(TranscriptionPipelineManager, "_shutdown")
@patch.object(TranscriptionPipelineManager, "_setup_signal_handlers")
def test_run_handles_idle_timeout(
    mock_setup_signals: MagicMock,
    mock_shutdown: MagicMock,
    mock_terminate_pods: MagicMock,
    mock_handle_wait_fail: MagicMock,
    mock_check_pod_idle: MagicMock,
    mock_handle_attempt_start: MagicMock,
    mock_handle_start_cycle: MagicMock,
    mock_get_current_time: MagicMock,
    mock_sleep: MagicMock,
    manager_instance: TranscriptionPipelineManager,
    mocker: MockerFixture,
) -> None:
    """Test exceeding STATUS_CHECK_TIMEOUT in WAITING_FOR_IDLE leads to failure state."""
    # Loop sequence:
    # 1: Start -> Attempt
    # 2: Attempt -> Wait
    # 3: Wait (Check=False, time < timeout) -> Wait
    # 4: Wait (Check=False, time >= timeout -> Terminate called, returns Fail state) -> Fail
    # 5: WaitFail -> WaitFail (Stop)
    num_loops = 5
    mocker.patch.object(manager_instance.shutdown_event, 'is_set', side_effect=[False] * num_loops + [True])

    # Simulate time: Start, small increments, then jump past STATUS_CHECK_TIMEOUT
    start_time = 1000.0
    time_before_timeout = start_time + const.STATUS_CHECK_TIMEOUT - 10
    time_after_timeout = start_time + const.STATUS_CHECK_TIMEOUT + 10
    mock_get_current_time.side_effect = [
        start_time,          # Loop 1 'now'
        start_time + 1,      # Loop 2 'now'
        time_before_timeout, # Loop 3 'now' (before timeout)
        time_after_timeout,  # Loop 4 'now' (after timeout)
        time_after_timeout + 1, # Loop 5 'now'
        time_after_timeout + 2, # Extra
    ]

    # --- Mock State Transitions ---
    mock_handle_start_cycle.return_value = (const.STATE_ATTEMPTING_POD_START, start_time, "", "", 0.0, 0.0, 0.0)
    mock_handle_attempt_start.return_value = (const.STATE_WAITING_FOR_IDLE, "pod", "url")
    # Mock the check called by the real _handle_waiting_for_idle to always return False (not idle)
    mock_check_pod_idle.return_value = False
    # Stay in failure state once reached
    mock_handle_wait_fail.return_value = const.STATE_WAITING_AFTER_FAILURE
    # --- End Mock ---

    # We need to let the *real* _handle_waiting_for_idle run to test its timeout logic
    with patch.object(manager_instance, '_handle_waiting_for_idle', wraps=manager_instance._handle_waiting_for_idle) as spy_handle_wait_idle:
        manager_instance.run() # Call directly

    # Assertions
    assert spy_handle_wait_idle.call_count >= 2 # Called in loop 3 and 4
    # Check idle status is called when interval is met (Loop 3 time vs Loop 2 time)
    # The check happens if now - last_idle_check >= interval.
    # Loop 3 now = time_before_timeout. last_idle_check initially 0. Interval met.
    # Loop 4 now = time_after_timeout. last_idle_check updated in loop 3 to time_before_timeout. Interval met.
    assert mock_check_pod_idle.call_count >= 1 # Should be called in loop 3 and maybe 4 depending on exact timing
    mock_terminate_pods.assert_called_once() # Called by handler in loop 4 due to timeout
    assert mock_handle_wait_fail.call_count >= 1 # Called in loop 5
    mock_shutdown.assert_called_once()


@patch.object(TranscriptionPipelineManager, "sleep", return_value=None)
@patch.object(TranscriptionPipelineManager, "get_current_time")
@patch.object(TranscriptionPipelineManager, "_handle_starting_cycle")
@patch.object(TranscriptionPipelineManager, "_handle_attempting_pod_start")
@patch.object(TranscriptionPipelineManager, "_handle_waiting_after_failure")
@patch.object(TranscriptionPipelineManager, "_shutdown")
@patch.object(TranscriptionPipelineManager, "_setup_signal_handlers")
def test_run_handles_pod_start_failure(
    mock_setup_signals: MagicMock,
    mock_shutdown: MagicMock,
    mock_handle_wait_fail: MagicMock,
    mock_handle_attempt_start: MagicMock,
    mock_handle_start_cycle: MagicMock,
    mock_get_current_time: MagicMock,
    mock_sleep: MagicMock,
    manager_instance: TranscriptionPipelineManager,
    mocker: MockerFixture,
) -> None:
    """Test failure during pod start transitions to WAITING_AFTER_FAILURE."""
    # Loop sequence:
    # 1: Start -> Attempt
    # 2: Attempt -> Fail
    # 3: WaitFail -> WaitFail (Stop)
    num_loops = 3
    mocker.patch.object(manager_instance.shutdown_event, 'is_set', side_effect=[False] * num_loops + [True])
    mock_get_current_time.side_effect = [1000.0 + i for i in range(num_loops + 1)]

    # --- Mock State Transitions ---
    start_time = 1000.0
    mock_handle_start_cycle.return_value = (const.STATE_ATTEMPTING_POD_START, start_time, "", "", 0.0, 0.0, 0.0)
    # Simulate failure in pod start handler
    mock_handle_attempt_start.return_value = (const.STATE_WAITING_AFTER_FAILURE, "", "")
    # Stay in failure state
    mock_handle_wait_fail.return_value = const.STATE_WAITING_AFTER_FAILURE
    # --- End Mock ---

    manager_instance.run() # Call directly

    # Assert handlers were called
    mock_handle_start_cycle.assert_called_once()
    mock_handle_attempt_start.assert_called_once()
    assert mock_handle_wait_fail.call_count >= 1
    mock_shutdown.assert_called_once()


@patch.object(TranscriptionPipelineManager, "sleep", return_value=None)
@patch.object(TranscriptionPipelineManager, "get_current_time")
@patch.object(TranscriptionPipelineManager, "_handle_starting_cycle")
@patch.object(TranscriptionPipelineManager, "_handle_attempting_pod_start")
@patch.object(TranscriptionPipelineManager, "_handle_waiting_for_idle")
@patch.object(TranscriptionPipelineManager, "_trigger_pipeline_run") # Mock the trigger it calls
@patch.object(TranscriptionPipelineManager, "_handle_waiting_after_failure")
@patch.object(TranscriptionPipelineManager, "_terminate_pods") # Called on failure
@patch.object(TranscriptionPipelineManager, "_shutdown")
@patch.object(TranscriptionPipelineManager, "_setup_signal_handlers")
def test_run_handles_pipeline_trigger_failure(
    mock_setup_signals: MagicMock,
    mock_shutdown: MagicMock,
    mock_terminate_pods: MagicMock,
    mock_handle_wait_fail: MagicMock,
    mock_trigger_pipeline: MagicMock,
    mock_handle_wait_idle: MagicMock,
    mock_handle_attempt_start: MagicMock,
    mock_handle_start_cycle: MagicMock,
    mock_get_current_time: MagicMock,
    mock_sleep: MagicMock,
    manager_instance: TranscriptionPipelineManager,
    mocker: MockerFixture,
) -> None:
    """Test failure during pipeline trigger transitions to WAITING_AFTER_FAILURE."""
    # Loop sequence:
    # 1: Start -> Attempt
    # 2: Attempt -> Wait
    # 3: Wait -> Run
    # 4: Run (Trigger=False -> Terminate called, returns Fail state) -> Fail
    # 5: WaitFail -> WaitFail (Stop)
    num_loops = 5
    mocker.patch.object(manager_instance.shutdown_event, 'is_set', side_effect=[False] * num_loops + [True])
    mock_get_current_time.side_effect = [1000.0 + i for i in range(num_loops + 1)]

    # --- Mock State Transitions ---
    start_time = 1000.0
    mock_handle_start_cycle.return_value = (const.STATE_ATTEMPTING_POD_START, start_time, "", "", 0.0, 0.0, 0.0)
    mock_handle_attempt_start.return_value = (const.STATE_WAITING_FOR_IDLE, "pod", "url")
    mock_handle_wait_idle.return_value = (const.STATE_ATTEMPTING_PIPELINE_RUN, start_time + 1)
    # Simulate failure in the trigger function called by the real handler
    mock_trigger_pipeline.return_value = False
    # Stay in failure state
    mock_handle_wait_fail.return_value = const.STATE_WAITING_AFTER_FAILURE
    # --- End Mock ---

    # Let the real _handle_attempting_pipeline_run execute
    with patch.object(manager_instance, '_handle_attempting_pipeline_run', wraps=manager_instance._handle_attempting_pipeline_run) as spy_handle_attempt_run:
        manager_instance.run() # Call directly

    # Assert handlers were called
    mock_handle_start_cycle.assert_called_once()
    mock_handle_attempt_start.assert_called_once()
    mock_handle_wait_idle.assert_called_once()
    spy_handle_attempt_run.assert_called_once() # Check the real handler ran
    mock_trigger_pipeline.assert_called_once() # Check the trigger was called by the real handler
    mock_terminate_pods.assert_called_once() # Check terminate was called by the real handler
    assert mock_handle_wait_fail.call_count >= 1 # Check we entered the failure state
    mock_shutdown.assert_called_once()


@patch.object(TranscriptionPipelineManager, "sleep", return_value=None)
@patch.object(TranscriptionPipelineManager, "get_current_time")
@patch.object(TranscriptionPipelineManager, "_handle_starting_cycle")
@patch.object(TranscriptionPipelineManager, "_handle_attempting_pod_start")
@patch.object(TranscriptionPipelineManager, "_handle_waiting_for_idle")
@patch.object(TranscriptionPipelineManager, "_handle_attempting_pipeline_run")
@patch.object(TranscriptionPipelineManager, "_handle_updating_counts")
@patch.object(TranscriptionPipelineManager, "_update_pod_counts") # Mock the helper
@patch.object(TranscriptionPipelineManager, "_shutdown")
@patch.object(TranscriptionPipelineManager, "_setup_signal_handlers")
def test_run_handles_count_update_interval(
    mock_setup_signals: MagicMock,
    mock_shutdown: MagicMock,
    mock_update_counts_helper: MagicMock,
    mock_handle_updating: MagicMock,
    mock_handle_attempt_run: MagicMock,
    mock_handle_wait_idle: MagicMock,
    mock_handle_attempt_start: MagicMock,
    mock_handle_start_cycle: MagicMock,
    mock_get_current_time: MagicMock,
    mock_sleep: MagicMock,
    manager_instance: TranscriptionPipelineManager,
    mocker: MockerFixture,
) -> None:
    """Test _update_pod_counts is called based on COUNT_UPDATE_INTERVAL."""
    # Loop sequence:
    # 1: Start -> Attempt
    # 2: Attempt -> Wait
    # 3: Wait -> Run
    # 4: Run -> Update
    # 5: Update (time < interval) -> Update
    # 6: Update (time >= interval) -> Update (calls helper)
    # 7: Update (time < interval) -> Update (Stop)
    num_loops = 7
    mocker.patch.object(manager_instance.shutdown_event, 'is_set', side_effect=[False] * num_loops + [True])

    # Simulate time: Start, small increments, then jump past COUNT_UPDATE_INTERVAL
    start_time = 1000.0
    time_before_interval = start_time + const.COUNT_UPDATE_INTERVAL - 10
    time_after_interval = start_time + const.COUNT_UPDATE_INTERVAL + 10
    mock_get_current_time.side_effect = [
        start_time,           # Loop 1
        start_time + 1,       # Loop 2
        start_time + 2,       # Loop 3
        start_time + 3,       # Loop 4
        time_before_interval, # Loop 5 (Update state starts, interval not met)
        time_after_interval,  # Loop 6 (Interval met)
        time_after_interval + 1, # Loop 7 (Interval not met again)
        time_after_interval + 2, # Extra
    ]

    # --- Mock State Transitions ---
    mock_handle_start_cycle.return_value = (const.STATE_ATTEMPTING_POD_START, start_time, "", "", 0.0, 0.0, 0.0)
    mock_handle_attempt_start.return_value = (const.STATE_WAITING_FOR_IDLE, "pod", "url")
    mock_handle_wait_idle.return_value = (const.STATE_ATTEMPTING_PIPELINE_RUN, start_time + 1)
    mock_handle_attempt_run.return_value = const.STATE_UPDATING_COUNTS # Enters UPDATE state in loop 5

    # Mock _handle_updating_counts with a side_effect to control interval logic
    # but force state to remain UPATING_COUNTS
    def handle_updating_side_effect(now, last_count_update_time_arg):
        next_state = const.STATE_UPDATING_COUNTS
        time_to_return = now # Initialize time on first entry

        if last_count_update_time_arg > 0.0: # Check only if initialized
            time_to_return = last_count_update_time_arg # Default to old time if initialized
            if now - last_count_update_time_arg >= const.COUNT_UPDATE_INTERVAL:
                mock_update_counts_helper()
                time_to_return = now # Update time only if interval met

        return next_state, time_to_return

    mock_handle_updating.side_effect = handle_updating_side_effect
    # Don't need return_value for helper if just checking call count
    # --- End Mock ---

    manager_instance.run() # Call directly

    # Assertions
    assert mock_handle_updating.call_count >= 3 # Loops 5, 6, 7
    # _update_pod_counts should only be called when interval is met (Loop 6)
    mock_update_counts_helper.assert_called_once()
    mock_shutdown.assert_called_once()


@patch.object(TranscriptionPipelineManager, "sleep", return_value=None)
@patch.object(TranscriptionPipelineManager, "get_current_time")
@patch.object(TranscriptionPipelineManager, "_handle_starting_cycle")
@patch.object(TranscriptionPipelineManager, "_handle_attempting_pod_start")
@patch.object(TranscriptionPipelineManager, "_handle_waiting_for_idle")
@patch.object(TranscriptionPipelineManager, "_check_pod_idle_status") # Mock the helper
@patch.object(TranscriptionPipelineManager, "_shutdown")
@patch.object(TranscriptionPipelineManager, "_setup_signal_handlers")
def test_run_handles_idle_check_interval(
    mock_setup_signals: MagicMock,
    mock_shutdown: MagicMock,
    mock_check_idle_helper: MagicMock,
    mock_handle_wait_idle: MagicMock,
    mock_handle_attempt_start: MagicMock,
    mock_handle_start_cycle: MagicMock,
    mock_get_current_time: MagicMock,
    mock_sleep: MagicMock,
    manager_instance: TranscriptionPipelineManager,
    mocker: MockerFixture,
) -> None:
    """Test _check_pod_idle_status is called based on POD_STATUS_CHECK_INTERVAL."""
    # Loop sequence:
    # 1: Start -> Attempt
    # 2: Attempt -> Wait
    # 3: Wait (time < interval) -> Wait
    # 4: Wait (time >= interval) -> Wait (calls helper)
    # 5: Wait (time < interval) -> Wait (Stop)
    num_loops = 5
    mocker.patch.object(manager_instance.shutdown_event, 'is_set', side_effect=[False] * num_loops + [True])

    # Simulate time: Start, small increments, then jump past POD_STATUS_CHECK_INTERVAL
    start_time = 1000.0
    time_before_interval = start_time + const.POD_STATUS_CHECK_INTERVAL - 1
    time_after_interval = start_time + const.POD_STATUS_CHECK_INTERVAL + 1
    mock_get_current_time.side_effect = [
        start_time,           # Loop 1
        start_time + 1,       # Loop 2
        time_before_interval, # Loop 3 (Wait state starts, interval not met)
        time_after_interval,  # Loop 4 (Interval met)
        time_after_interval + 1, # Loop 5 (Interval not met again)
        time_after_interval + 2, # Extra
    ]

    # --- Mock State Transitions ---
    mock_handle_start_cycle.return_value = (const.STATE_ATTEMPTING_POD_START, start_time, "", "", 0.0, 0.0, 0.0)
    mock_handle_attempt_start.return_value = (const.STATE_WAITING_FOR_IDLE, "pod", "url") # Enters WAIT state in loop 3

    # Mock _handle_waiting_for_idle with a side_effect to control interval logic
    # but force state to remain WAITING_FOR_IDLE
    def handle_wait_idle_side_effect(now, elapsed, last_idle_check_time_arg, *args):
        next_state = const.STATE_WAITING_FOR_IDLE
        time_to_return = now # Initialize time on first entry

        if last_idle_check_time_arg > 0.0: # Check only if initialized
            time_to_return = last_idle_check_time_arg # Default to old time if initialized
            if now - last_idle_check_time_arg >= const.POD_STATUS_CHECK_INTERVAL:
                _ = mock_check_idle_helper(*args[-1:]) # Pass pod_url
                time_to_return = now # Update time only if interval met

        return next_state, time_to_return

    mock_handle_wait_idle.side_effect = handle_wait_idle_side_effect
    # Don't need return_value for helper if just checking call count
    # --- End Mock ---

    manager_instance.run() # Call directly

    # Assertions
    assert mock_handle_wait_idle.call_count >= 3 # Loops 3, 4, 5
    # _check_pod_idle_status should only be called when interval is met (Loop 4)
    mock_check_idle_helper.assert_called_once()
    mock_shutdown.assert_called_once()


@patch.object(TranscriptionPipelineManager, "sleep", return_value=None)
@patch.object(TranscriptionPipelineManager, "get_current_time")
@patch.object(TranscriptionPipelineManager, "_handle_starting_cycle")
@patch.object(TranscriptionPipelineManager, "_shutdown")
@patch.object(TranscriptionPipelineManager, "_setup_signal_handlers")
def test_run_raises_runtimeerror_on_invalid_state(
    mock_setup_signals: MagicMock,
    mock_shutdown: MagicMock,
    mock_handle_start_cycle: MagicMock,
    mock_get_current_time: MagicMock,
    mock_sleep: MagicMock,
    manager_instance: TranscriptionPipelineManager,
    mock_logger: MagicMock, # Use logger to check for critical error
    mocker: MockerFixture,
) -> None:
    """Test run() catches RuntimeError, logs critical, and shuts down."""
    shutdown_event = manager_instance.shutdown_event
    start_time = 1000.0
    invalid_state = "INVALID_STATE_XYZ"

    # --- Mock State Transitions ---
    mock_handle_start_cycle.return_value = (invalid_state, start_time, "", "", 0.0, 0.0, 0.0)
    # --- End Mock ---

    # Mock time for the first loop iteration
    mock_get_current_time.return_value = start_time
    # DO NOT mock shutdown_event.is_set - let the exception handler set the real event

    # Expect RuntimeError to be caught by the main try/except in run()
    # which logs critically and calls shutdown
    manager_instance.run() # Call directly

    # Check that the critical log was called (verifies exception handling path)
    mock_logger.critical.assert_called()

    # Check that the shutdown event was set by the exception handler
    assert shutdown_event.is_set()
    mock_shutdown.assert_called_once()


# _shutdown
def test_shutdown_calls_rest_interface_shutdown(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _shutdown calls shutdown on the RestInterface instance."""
    manager_instance._shutdown()

    mock_rest_interface.shutdown.assert_called_once()
    mock_logger.debug.assert_any_call("REST interface shutdown complete.")


def test_shutdown_handles_no_rest_interface(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _shutdown handles the case where RestInterface was not initialized."""
    manager_instance.rest_interface = None
    manager_instance._shutdown()
    mock_logger.debug.assert_any_call("REST interface shutdown complete.")


def test_shutdown_handles_rest_interface_exception(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _shutdown logs an error if RestInterface.shutdown() raises an exception."""
    test_exception = Exception("Flask server error on shutdown")
    mock_rest_interface.shutdown.side_effect = test_exception

    manager_instance._shutdown()

    mock_rest_interface.shutdown.assert_called_once()
    mock_logger.error.assert_called_once_with(
        f"Error shutting down REST interface: {test_exception}", exc_info=False
    )
    mock_logger.debug.assert_any_call("REST interface shutdown complete.")

# _terminate_pods
def test_terminate_pods_success(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_runpod_manager: tuple[MagicMock, MagicMock]) -> None:
    """Test _terminate_pods calls the terminate manager and logs success."""
    _, mock_terminate_manager = mock_runpod_manager
    mock_terminate_manager.run.return_value = True

    manager_instance._terminate_pods()

    mock_terminate_manager.run.assert_called_once()
    mock_logger.info.assert_any_call("Pod termination/stop process completed.")


def test_terminate_pods_failure(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_runpod_manager: tuple[MagicMock, MagicMock]) -> None:
    """Test _terminate_pods calls the terminate manager and logs failure."""
    _, mock_terminate_manager = mock_runpod_manager
    mock_terminate_manager.run.return_value = None

    manager_instance._terminate_pods()

    mock_terminate_manager.run.assert_called_once()
    mock_logger.error.assert_any_call("Pod termination/stop process did not complete successfully.")


def test_terminate_pods_exception(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_runpod_manager: tuple[MagicMock, MagicMock]) -> None:
    """Test _terminate_pods catches exceptions from the terminate manager."""
    _, mock_terminate_manager = mock_runpod_manager
    test_exception = Exception("RunPod API error")
    mock_terminate_manager.run.side_effect = test_exception

    manager_instance._terminate_pods()

    mock_terminate_manager.run.assert_called_once()
    mock_logger.error.assert_called_once_with(
        f"An error occurred during pod termination/stop: {test_exception}", exc_info=False
    )


# _update_pod_counts
def test_update_pod_counts_success(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_runpod_manager: tuple[MagicMock, MagicMock], mock_rest_interface: MagicMock) -> None:
    """Test _update_pod_counts successfully gets counts and updates REST interface."""
    mock_start_manager, _ = mock_runpod_manager
    counts = {'total': 5, 'running': 3}
    mock_start_manager.count_pods.return_value = counts

    manager_instance._update_pod_counts()

    mock_start_manager.count_pods.assert_called_once()
    mock_rest_interface.update_pods_total.assert_called_once_with(5)
    mock_rest_interface.update_pods_running.assert_called_once_with(3)


def test_update_pod_counts_exception(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_runpod_manager: tuple[MagicMock, MagicMock], mock_rest_interface: MagicMock) -> None:
    """Test _update_pod_counts handles exceptions from count_pods."""
    mock_start_manager, _ = mock_runpod_manager
    test_exception = Exception("RunPod API error during count")
    mock_start_manager.count_pods.side_effect = test_exception

    manager_instance._update_pod_counts()

    mock_start_manager.count_pods.assert_called_once()
    mock_logger.error.assert_called_once_with(
        f"Failed to retrieve pod counts: {test_exception}", exc_info=False
    )
    mock_rest_interface.update_pods_total.assert_not_called()
    mock_rest_interface.update_pods_running.assert_not_called()


@pytest.mark.parametrize("invalid_counts", [
    {'total': 5},
    {'running': 3},
    {},
    "not a dict",
])
def test_update_pod_counts_invalid_dict(invalid_counts: Any, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_runpod_manager: tuple[MagicMock, MagicMock], mock_rest_interface: MagicMock) -> None:
    """Test _update_pod_counts handles invalid dictionary format from count_pods."""
    mock_start_manager, _ = mock_runpod_manager
    mock_start_manager.count_pods.return_value = invalid_counts

    manager_instance._update_pod_counts()

    mock_start_manager.count_pods.assert_called_once()
    mock_logger.warning.assert_called_once_with("Did not receive valid pod counts from RunpodSingletonManager.")
    mock_rest_interface.update_pods_total.assert_not_called()
    mock_rest_interface.update_pods_running.assert_not_called()


# _setup_signal_handlers & _handle_shutdown_signal
@patch("transcription_pipeline_manager.manager.signal.signal")
def test_setup_signal_handlers(mock_signal: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _setup_signal_handlers registers handlers for SIGINT and SIGTERM."""
    manager_instance._setup_signal_handlers()

    expected_calls = [
        call(signal.SIGINT, manager_instance._handle_shutdown_signal),
        call(signal.SIGTERM, manager_instance._handle_shutdown_signal),
    ]
    mock_signal.assert_has_calls(expected_calls, any_order=True)
    assert mock_signal.call_count == 2


def test_handle_shutdown_signal(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _handle_shutdown_signal sets the shutdown event and logs."""
    assert not manager_instance.shutdown_event.is_set() # Ensure it's initially False
    manager_instance._handle_shutdown_signal(signal.SIGINT, None)
    assert manager_instance.shutdown_event.is_set()
    manager_instance.shutdown_event.clear()
    mock_logger.reset_mock()
    manager_instance._handle_shutdown_signal(signal.SIGTERM, None)
    assert manager_instance.shutdown_event.is_set()


@patch("transcription_pipeline_manager.manager.post_request")
def test_trigger_pipeline_run_retry_error(mock_post_request: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _trigger_pipeline_run logs correctly when post_request raises RetryError."""
    request_exception = RequestException("Connection failed")
    # Mock post_request itself to raise the RetryError containing the original exception
    # Note: In a real scenario, the @retry decorator handles this wrapping.
    # Here, we simulate the outcome of the retries failing.
    mock_post_request.side_effect = RetryError(last_attempt=MagicMock(exception=lambda: request_exception))
    result = manager_instance._trigger_pipeline_run(TEST_POD_URL)
    assert result is False
    mock_post_request.assert_called_once()
    mock_logger.error.assert_called_once_with(
        f"Failed to trigger pipeline run at {TEST_POD_URL}/run after multiple retries.",
        exc_info=True
    )


@patch("transcription_pipeline_manager.manager.post_request")
@patch.object(TranscriptionPipelineManager, "_process_trigger_pipeline_run_response")
def test_trigger_pipeline_run_processing_exception(mock_process_response: MagicMock, mock_post_request: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _trigger_pipeline_run handles exceptions during response processing."""
    mock_post_request.return_value = MagicMock(spec=requests.Response)
    processing_exception = ValueError("Processing failed")
    mock_process_response.side_effect = processing_exception
    result = manager_instance._trigger_pipeline_run(TEST_POD_URL)
    assert result is False
    mock_post_request.assert_called_once()
    mock_process_response.assert_called_once()
    mock_logger.error.assert_called_once_with(
        f"An unexpected error occurred triggering pipeline run at {TEST_POD_URL}/run: {processing_exception}",
        exc_info=manager_instance.debug
    )
    mock_rest_interface.update_pipeline_last_run_time.assert_not_called()


@pytest.mark.parametrize(
    "ngrok_enabled, expected_url_pattern",
    [
        (False, f"https://www.{TEST_DOMAIN}/api/transcription/logs?api_key={TEST_API_KEY}"),
        (True, f"{const.NGROK_DOMAIN}/logs?api_key={TEST_API_KEY}"),
    ],
    ids=["ngrok_disabled", "ngrok_enabled"]
)
@patch("transcription_pipeline_manager.manager.Logger")
@patch("transcription_pipeline_manager.manager.RestInterface")
@patch("transcription_pipeline_manager.manager.RunpodSingletonManager")
@patch("transcription_pipeline_manager.manager.Path")
def test_build_logs_callback_url(
    mock_path: MagicMock,
    mock_runpod: MagicMock,
    mock_rest: MagicMock,
    mock_logger: MagicMock,
    ngrok_enabled: bool,
    expected_url_pattern: str
) -> None:
    """Test _build_logs_callback_url constructs the correct URL based on ngrok flag."""
    # Minimal setup needed just for this method
    manager = TranscriptionPipelineManager(
        api_key=TEST_API_KEY,
        domain=TEST_DOMAIN,
        ngrok=ngrok_enabled,
        debug=False,
    )
    actual_url = manager._build_logs_callback_url()
    assert actual_url == expected_url_pattern
