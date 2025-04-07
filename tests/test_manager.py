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
            api_key=TEST_API_KEY,
            debug=False,
        ),
        call(
            config_path=expected_config_path,
            api_key=TEST_API_KEY,
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
    custom_debug = True

    manager = TranscriptionPipelineManager(
        api_key=TEST_API_KEY,
        domain=TEST_DOMAIN,
        limit=custom_limit,
        processing_limit=custom_processing_limit,
        debug=custom_debug,
    )

    assert manager.api_key == TEST_API_KEY
    assert manager.domain == TEST_DOMAIN
    assert manager.limit == custom_limit
    assert manager.processing_limit == custom_processing_limit
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
            api_key=TEST_API_KEY,
            debug=custom_debug,
        ),
        call(
            config_path=expected_config_path,
            api_key=TEST_API_KEY,
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
    mock_response.status_code = 500
    mock_response.reason = "Internal Server Error"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mock_get.return_value = mock_response

    result = manager_instance._check_pod_idle_status(TEST_POD_URL)

    assert result is False
    mock_get.assert_called_once_with(
        f"{TEST_POD_URL}/status",
        timeout=const.POD_REQUEST_TIMEOUT
    )
    mock_logger.warning.assert_called_once_with(
        f"Failed to get pod status from {TEST_POD_URL}/status. Status: 500 Internal Server Error"
    )


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
    mock_logger.warning.assert_called_once_with(
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
@patch("transcription_pipeline_manager.manager.requests.post")
@patch("transcription_pipeline_manager.manager.time.time", return_value=1234567890)
def test_trigger_pipeline_run_success(mock_time: MagicMock, mock_post: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _trigger_pipeline_run returns True on successful API call."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"message": "Pipeline run triggered successfully"}
    mock_post.return_value = mock_response

    expected_payload = {
        "api_key": TEST_API_KEY,
        "domain": manager_instance.transcription_subdomain,
        "limit": manager_instance.limit,
        "processing_limit": manager_instance.processing_limit,
        "callback_url": manager_instance.logs_callback_url,
    }

    result = manager_instance._trigger_pipeline_run(TEST_POD_URL)

    assert result is True
    mock_post.assert_called_once_with(
        f"{TEST_POD_URL}/run",
        json=expected_payload,
        timeout=const.POD_REQUEST_TIMEOUT
    )
    mock_rest_interface.update_pipeline_last_run_time.assert_called_once_with(1234567890)


@patch("transcription_pipeline_manager.manager.requests.post")
def test_trigger_pipeline_run_api_error(mock_post: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _trigger_pipeline_run returns False when API returns an error message."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"error": "Invalid parameters"}
    mock_post.return_value = mock_response

    result = manager_instance._trigger_pipeline_run(TEST_POD_URL)

    assert result is False
    mock_post.assert_called_once()
    mock_logger.error.assert_called_once_with(
        f"Pipeline run trigger failed at {TEST_POD_URL}/run: Invalid parameters"
    )
    mock_rest_interface.update_pipeline_last_run_time.assert_not_called()


# _shutdown
def test_shutdown_calls_rest_interface_shutdown(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _shutdown calls shutdown on the RestInterface instance."""
    manager_instance._shutdown()

    mock_rest_interface.shutdown.assert_called_once()
    mock_logger.info.assert_any_call("REST interface shutdown complete.")


def test_shutdown_handles_no_rest_interface(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock) -> None:
    """Test _shutdown handles the case where RestInterface was not initialized."""
    manager_instance.rest_interface = None
    manager_instance._shutdown()
    mock_logger.info.assert_any_call("REST interface shutdown complete.")


def test_shutdown_handles_rest_interface_exception(manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _shutdown logs an error if RestInterface.shutdown() raises an exception."""
    test_exception = Exception("Flask server error on shutdown")
    mock_rest_interface.shutdown.side_effect = test_exception

    manager_instance._shutdown()

    mock_rest_interface.shutdown.assert_called_once()
    mock_logger.error.assert_called_once_with(
        f"Error shutting down REST interface: {test_exception}", exc_info=False
    )
    mock_logger.info.assert_any_call("REST interface shutdown complete.")

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
    mock_logger.warning.assert_any_call("Pod termination/stop process did not complete successfully.")


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


@patch("transcription_pipeline_manager.manager.requests.post")
def test_trigger_pipeline_run_request_exception(mock_post: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _trigger_pipeline_run returns False on RequestException."""
    mock_post.side_effect = RequestException("Connection failed")

    result = manager_instance._trigger_pipeline_run(TEST_POD_URL)

    assert result is False
    mock_post.assert_called_once()
    mock_logger.error.assert_called_once_with(
        f"Error triggering pipeline run at {TEST_POD_URL}/run: Connection failed", exc_info=False
    )
    mock_rest_interface.update_pipeline_last_run_time.assert_not_called()


@patch("transcription_pipeline_manager.manager.requests.post")
def test_trigger_pipeline_run_http_error(mock_post: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _trigger_pipeline_run returns False on non-200 status code."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 503
    mock_response.reason = "Service Unavailable"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mock_post.return_value = mock_response

    result = manager_instance._trigger_pipeline_run(TEST_POD_URL)

    assert result is False
    mock_post.assert_called_once()
    mock_logger.warning.assert_called_once_with(
        f"Failed to trigger pipeline run at {TEST_POD_URL}/run. Status: 503 Service Unavailable"
    )
    mock_rest_interface.update_pipeline_last_run_time.assert_not_called()


@patch("transcription_pipeline_manager.manager.requests.post")
def test_trigger_pipeline_run_json_decode_error(mock_post: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _trigger_pipeline_run returns False on JSONDecodeError."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    json_exception = json.JSONDecodeError("Expecting value", "doc", 0)
    mock_response.json.side_effect = json_exception
    mock_post.return_value = mock_response

    result = manager_instance._trigger_pipeline_run(TEST_POD_URL)

    assert result is False
    mock_post.assert_called_once()
    mock_logger.warning.assert_called_once_with(
        f"Failed to decode JSON response from {TEST_POD_URL}/run: {json_exception}.", exc_info=False
    )
    mock_rest_interface.update_pipeline_last_run_time.assert_not_called()


@patch("transcription_pipeline_manager.manager.requests.post")
def test_trigger_pipeline_run_unexpected_json(mock_post: MagicMock, manager_instance: TranscriptionPipelineManager, mock_logger: MagicMock, mock_rest_interface: MagicMock) -> None:
    """Test _trigger_pipeline_run returns False on unexpected JSON structure."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    json_return_value = {"bad": "values"}
    mock_response.json.return_value = json_return_value
    mock_post.return_value = mock_response

    result = manager_instance._trigger_pipeline_run(TEST_POD_URL)

    assert result is False
    mock_post.assert_called_once()
    mock_logger.warning.assert_called_once_with(
        f"Unexpected JSON response format from {TEST_POD_URL}/run: {json_return_value}"
    )
    mock_rest_interface.update_pipeline_last_run_time.assert_not_called()
