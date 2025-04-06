#!/usr/bin/env python3
"""
Tests for the transcription_pipeline_manager.rest_interface module.
"""

import argparse
import logging
import os
import threading
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from flask.testing import FlaskClient

from transcription_pipeline_manager import constants as const
from transcription_pipeline_manager.rest_interface import (
    CallbackServer,
    Stats,
    main,
    parse_args,
)


TEST_HOST = "127.0.0.1"
TEST_PORT = 5001
TEST_API_KEY = "test-secret-key"
TEST_TIMESTAMP = 1234567890


@pytest.fixture
def stats() -> Stats:
    """Fixture for a Stats instance."""
    return Stats()


@pytest.fixture(params=[False, True], ids=["nodebug", "debug"])
def debug_mode(request) -> bool:
    """Fixture to parameterize tests for debug mode."""
    return request.param


@pytest.fixture
def server_no_api_key(debug_mode) -> CallbackServer:
    """Fixture for a CallbackServer instance without an API key."""
    original_env_key = os.environ.pop("TRANSCRIPTION_API_KEY", None)
    server = CallbackServer(host=TEST_HOST, port=TEST_PORT, api_key=None, debug=debug_mode)
    server._register_routes()
    yield server
    if original_env_key is not None:
        os.environ["TRANSCRIPTION_API_KEY"] = original_env_key
    if server.thread and server.thread.is_alive():
        server.shutdown()


@pytest.fixture
def client_no_api_key(server_no_api_key: CallbackServer) -> FlaskClient:
    """Fixture for a test client for a server without an API key."""
    return server_no_api_key.app.test_client()


@pytest.fixture
def server_with_api_key(debug_mode) -> CallbackServer:
    """Fixture for a CallbackServer instance with an API key."""
    server = CallbackServer(host=TEST_HOST, port=TEST_PORT, api_key=TEST_API_KEY, debug=debug_mode)
    server._register_routes()
    yield server
    if server.thread and server.thread.is_alive():
        server.shutdown()


@pytest.fixture
def client_with_api_key(server_with_api_key: CallbackServer) -> FlaskClient:
    """Fixture for a test client for a server with an API key."""
    return server_with_api_key.app.test_client()



def test_stats_initialization(stats: Stats) -> None:
    """Test Stats default initialization."""
    assert stats.get_pods_total() == 0
    assert stats.get_pods_running() == 0
    assert stats.get_pipeline_last_run_time() == 0


def test_stats_set_get_pods_total(stats: Stats) -> None:
    """Test setting and getting pods_total."""
    stats.set_pods_total(5)
    assert stats.get_pods_total() == 5
    stats.set_pods_total(0)
    assert stats.get_pods_total() == 0


def test_stats_set_get_pipeline_last_run_time(stats: Stats) -> None:
    """Test setting and getting pipeline_last_run_time."""
    stats.set_pipeline_last_run_time(TEST_TIMESTAMP)
    assert stats.get_pipeline_last_run_time() == TEST_TIMESTAMP
    stats.set_pipeline_last_run_time(0)
    assert stats.get_pipeline_last_run_time() == 0


def test_stats_set_get_pods_running(stats: Stats) -> None:
    """Test setting and getting pods_running."""
    stats.set_pods_running(3)
    assert stats.get_pods_running() == 3
    stats.set_pods_running(0)
    assert stats.get_pods_running() == 0


def test_stats_get_all_stats(stats: Stats) -> None:
    """Test the get_all_stats method."""
    initial_stats = stats.get_all_stats()
    assert initial_stats == {
        "pods_total": 0,
        "pods_running": 0,
        "pipeline_last_run_time": 0,
    }

    stats.set_pods_total(5)
    stats.set_pods_running(3)
    stats.set_pipeline_last_run_time(TEST_TIMESTAMP)

    updated_stats = stats.get_all_stats()
    assert updated_stats == {
        "pods_total": 5,
        "pods_running": 3,
        "pipeline_last_run_time": TEST_TIMESTAMP,
    }



def test_server_init_defaults(debug_mode: bool) -> None:
    """Test CallbackServer initialization with default arguments."""
    original_env_key = os.environ.pop("TRANSCRIPTION_API_KEY", None)
    server = CallbackServer(host=TEST_HOST, port=TEST_PORT, debug=debug_mode)
    assert server.host == TEST_HOST
    assert server.port == TEST_PORT
    assert server.api_key is None
    assert server.debug == debug_mode
    assert isinstance(server.log, logging.Logger)
    assert isinstance(server.app, Flask)
    assert isinstance(server.stats, Stats)
    assert server.thread is None
    assert isinstance(server.stop_event, threading.Event)
    if original_env_key is not None:
        os.environ["TRANSCRIPTION_API_KEY"] = original_env_key


def test_server_init_custom(debug_mode: bool) -> None:
    """Test CallbackServer initialization with custom arguments."""
    custom_host = "0.0.0.0"
    custom_port = 9999
    custom_key = "custom-key"
    server = CallbackServer(host=custom_host, port=custom_port, api_key=custom_key, debug=debug_mode)
    assert server.host == custom_host
    assert server.port == custom_port
    assert server.api_key == custom_key
    assert server.debug == debug_mode


def test_server_init_api_key_env_var(monkeypatch, debug_mode: bool) -> None:
    """Test CallbackServer initialization picks up API key from environment variable."""
    monkeypatch.setenv("TRANSCRIPTION_API_KEY", "env-key")
    server = CallbackServer(host=TEST_HOST, port=TEST_PORT, api_key=None, debug=debug_mode)
    assert server.api_key == "env-key"
    server_override = CallbackServer(host=TEST_HOST, port=TEST_PORT, api_key="override-key", debug=debug_mode)
    assert server_override.api_key == "override-key"


def test_api_key_required_endpoints(client_with_api_key: FlaskClient) -> None:
    """Test that endpoints require API key when server is configured with one."""
    endpoints = ["/stats", "/logs"]
    for endpoint in endpoints:
        method = "POST" if endpoint == "/logs" else "GET"
        response = client_with_api_key.open(endpoint, method=method)
        assert response.status_code == 403
        assert response.json == {"error": "Forbidden", "message": "API key required"}


def test_api_key_invalid(client_with_api_key: FlaskClient) -> None:
    """Test that endpoints reject invalid API keys."""
    endpoints = ["/stats", "/logs"]
    for endpoint in endpoints:
        method = "POST" if endpoint == "/logs" else "GET"
        response = client_with_api_key.open(f"{endpoint}?api_key=wrongkey", method=method)
        assert response.status_code == 403
        assert response.json == {"error": "Forbidden", "message": "Invalid API key"}


def test_api_key_correct(client_with_api_key: FlaskClient) -> None:
    """Test that endpoints accept the correct API key."""
    response = client_with_api_key.get(f"/stats?api_key={TEST_API_KEY}")
    assert response.status_code == 200

    response = client_with_api_key.post(f"/logs?api_key={TEST_API_KEY}", data="test data")
    assert response.status_code == 200


def test_api_key_not_required(client_no_api_key: FlaskClient) -> None:
    """Test that endpoints are accessible without API key if server is not configured."""
    response = client_no_api_key.get("/stats")
    assert response.status_code == 200

    response = client_no_api_key.post("/logs", data="test data")
    assert response.status_code == 200


def test_get_stats_initial(client_no_api_key: FlaskClient) -> None:
    """Test GET /stats initial value."""
    response = client_no_api_key.get("/stats")
    assert response.status_code == 200
    assert response.json == {
        "stats": {"pods_total": 0, "pods_running": 0, "pipeline_last_run_time": 0}
    }


def test_get_stats_updated(server_no_api_key: CallbackServer, client_no_api_key: FlaskClient) -> None:
    """Test GET /stats after updating values."""
    server_no_api_key.update_pods_total(10)
    server_no_api_key.update_pods_running(5)
    server_no_api_key.update_pipeline_last_run_time(TEST_TIMESTAMP)
    response = client_no_api_key.get("/stats")
    assert response.status_code == 200
    assert response.json == {
        "stats": {"pods_total": 10, "pods_running": 5, "pipeline_last_run_time": TEST_TIMESTAMP}
    }


def test_get_stats_error(mocker, server_no_api_key: CallbackServer, client_no_api_key: FlaskClient) -> None:
    """Test GET /stats when an internal error occurs."""
    mocker.patch.object(server_no_api_key.stats, "get_all_stats", side_effect=Exception("Test error"))
    response = client_no_api_key.get("/stats")
    assert response.status_code == 500
    assert response.json == {"error": "Internal Server Error"}


def test_post_logs_json(mocker, server_no_api_key: CallbackServer, client_no_api_key: FlaskClient) -> None:
    """Test POST /logs with valid JSON data."""
    mock_log = mocker.patch.object(server_no_api_key, "log")
    json_data = {"key1": "value1", "status": "completed"}
    response = client_no_api_key.post("/logs", json=json_data)
    assert response.status_code == 200
    assert response.text == "OK"
    mock_log.info.assert_any_call("Received POST request to: /logs")
    mock_log.debug.assert_any_call("Content-Type: application/json")
    mock_log.debug.assert_any_call("Received JSON payload")


def test_post_logs_invalid_json(mocker, server_no_api_key: CallbackServer, client_no_api_key: FlaskClient) -> None:
    """Test POST /logs with invalid JSON data but correct Content-Type."""
    mock_log = mocker.patch.object(server_no_api_key, "log")
    invalid_json_string = '{"key": "value", "malformed": }'
    response = client_no_api_key.post(
        "/logs",
        data=invalid_json_string,
        content_type="application/json"
    )
    assert response.status_code == 200
    assert response.text == "OK"
    mock_log.warning.assert_any_call("Content-Type is application/json but failed to parse body.")
    mock_log.info.assert_any_call(f"Raw body:\n{invalid_json_string}")


def test_post_logs_non_json(mocker, server_no_api_key: CallbackServer, client_no_api_key: FlaskClient) -> None:
    """Test POST /logs with non-JSON data."""
    mock_log = mocker.patch.object(server_no_api_key, "log")
    text_data = "This is plain text data."
    response = client_no_api_key.post(
        "/logs",
        data=text_data,
        content_type="text/plain"
    )
    assert response.status_code == 200
    assert response.text == "OK"
    mock_log.debug.assert_any_call("Content-Type: text/plain")
    mock_log.info.assert_any_call(f"Received non-JSON body:\n{text_data}")


def test_post_logs_error(mocker, server_no_api_key: CallbackServer, client_no_api_key: FlaskClient) -> None:
    """Test POST /logs when an internal error occurs during processing."""
    mock_log = mocker.patch.object(server_no_api_key, "log")
    test_exception = Exception("Test data read error")
    # Mock get_data to raise an exception
    mocker.patch("flask.Request.get_data", side_effect=test_exception)
    response = client_no_api_key.post("/logs", data="anything")
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    mock_log.error.assert_called_once_with(
        f"Error processing request to /logs: {test_exception}",
        exc_info=server_no_api_key.debug
    )

def test_update_pods_total(mocker, server_no_api_key: CallbackServer) -> None:
    """Test the update_pods_total method."""
    mock_log = mocker.patch.object(server_no_api_key, "log")
    server_no_api_key.update_pods_total(10)
    assert server_no_api_key.stats.get_pods_total() == 10
    mock_log.debug.assert_called_once_with("Updating pods_total to 10")


def test_update_pods_running(mocker, server_no_api_key: CallbackServer) -> None:
    """Test the update_pods_running method."""
    mock_log = mocker.patch.object(server_no_api_key, "log")
    server_no_api_key.update_pods_running(7)
    assert server_no_api_key.stats.get_pods_running() == 7
    mock_log.debug.assert_called_once_with("Updating pods_running to 7")


def test_update_pipeline_last_run_time(mocker, server_no_api_key: CallbackServer) -> None:
    """Test the update_pipeline_last_run_time method."""
    mock_log = mocker.patch.object(server_no_api_key, "log")
    server_no_api_key.update_pipeline_last_run_time(TEST_TIMESTAMP + 1)
    assert server_no_api_key.stats.get_pipeline_last_run_time() == TEST_TIMESTAMP + 1
    mock_log.debug.assert_called_once_with(f"Updating pipeline_last_run_time to {TEST_TIMESTAMP + 1}")



@patch("threading.Thread")
@patch("transcription_pipeline_manager.rest_interface.CallbackServer._register_routes")
@patch("flask.Flask.run")
def test_server_start(mock_flask_run, mock_register_routes, mock_thread_class, server_no_api_key: CallbackServer, mocker) -> None:
    """Test the start method of CallbackServer."""
    mock_thread_instance = MagicMock()
    mock_thread_class.return_value = mock_thread_instance

    server_no_api_key.start()

    mock_register_routes.assert_called_once()
    mock_thread_class.assert_called_once_with(target=mocker.ANY, daemon=True)
    thread_args, thread_kwargs = mock_thread_class.call_args
    target_func = thread_kwargs.get('target') or thread_args[0]
    assert target_func is not None

    target_func()
    mock_flask_run.assert_called_once_with(host=TEST_HOST, port=TEST_PORT, debug=False, use_reloader=False)

    mock_thread_instance.start.assert_called_once()
    assert server_no_api_key.thread == mock_thread_instance


def test_server_shutdown_running(server_no_api_key: CallbackServer) -> None:
    """Test shutdown when the server thread is running."""
    mock_thread_instance = MagicMock()
    mock_thread_instance.is_alive.return_value = True
    server_no_api_key.thread = mock_thread_instance

    server_no_api_key.shutdown()

    assert server_no_api_key.stop_event.is_set()
    mock_thread_instance.join.assert_called_once_with(timeout=2.0)
    assert server_no_api_key.thread is None


def test_server_shutdown_not_running(server_no_api_key: CallbackServer) -> None:
    """Test shutdown when the server thread is not running or already finished."""
    mock_thread_instance = MagicMock()
    mock_thread_instance.is_alive.return_value = False
    server_no_api_key.thread = mock_thread_instance

    server_no_api_key.shutdown()

    assert server_no_api_key.stop_event.is_set()
    mock_thread_instance.join.assert_not_called()
    assert server_no_api_key.thread is None


def test_server_shutdown_no_thread(server_no_api_key: CallbackServer) -> None:
    """Test shutdown when the server thread was never started."""
    assert server_no_api_key.thread is None
    server_no_api_key.shutdown()
    assert server_no_api_key.stop_event.is_set()
    assert server_no_api_key.thread is None



def test_parse_args_defaults(mocker) -> None:
    """Test parse_args with default values."""
    mocker.patch("sys.argv", ["script_name"])
    args = parse_args()
    assert args.host == const.DEFAULT_REST_HOST
    assert args.port == const.DEFAULT_REST_PORT
    assert args.api_key is None
    assert args.debug is False


def test_parse_args_custom(mocker) -> None:
    """Test parse_args with custom arguments."""
    custom_host = "192.168.1.100"
    custom_port = "9000"
    custom_key = "cli-key"
    mocker.patch(
        "sys.argv",
        [
            "script_name",
            "--host", custom_host,
            "--port", custom_port,
            "--api-key", custom_key,
            "--debug",
        ],
    )
    args = parse_args()
    assert args.host == custom_host
    assert args.port == int(custom_port)
    assert args.api_key == custom_key
    assert args.debug is True



@patch("transcription_pipeline_manager.rest_interface.parse_args")
@patch("transcription_pipeline_manager.rest_interface.CallbackServer")
@patch("transcription_pipeline_manager.rest_interface.Logger")
@patch("time.sleep", side_effect=KeyboardInterrupt)
@patch("sys.exit")
def test_main_success_flow(mock_exit, mock_sleep, mock_logger_class, mock_server_class, mock_parse_args) -> None:
    """Test the main function's normal execution flow with KeyboardInterrupt."""
    mock_args = argparse.Namespace(
        host="main_host", port=1234, api_key="main_key", debug=True
    )
    mock_parse_args.return_value = mock_args
    mock_server_instance = MagicMock(spec=CallbackServer)
    mock_server_instance.stop_event = threading.Event()
    mock_server_class.return_value = mock_server_instance
    mock_log_instance = MagicMock()
    mock_logger_class.return_value = mock_log_instance

    main()

    mock_parse_args.assert_called_once()
    mock_logger_class.assert_called_with("transcription_pipeline_manager.rest_interface", debug=True)
    mock_server_class.assert_called_once_with(
        host="main_host", port=1234, api_key="main_key", debug=True
    )
    mock_server_instance.start.assert_called_once()
    mock_sleep.assert_called_once_with(1)
    mock_log_instance.info.assert_any_call("Shutdown signal received (KeyboardInterrupt).")
    mock_server_instance.shutdown.assert_called_once()
    mock_exit.assert_called_once_with(const.EXIT_SUCCESS)


@patch("transcription_pipeline_manager.rest_interface.parse_args")
@patch("transcription_pipeline_manager.rest_interface.CallbackServer")
@patch("transcription_pipeline_manager.rest_interface.Logger")
@patch("traceback.print_exc")
@patch("sys.exit")
def test_main_exception_flow(mock_exit, mock_print_exc, mock_logger_class, mock_server_class, mock_parse_args) -> None:
    """Test the main function's flow when an exception occurs during setup."""
    mock_args = argparse.Namespace(
        host="main_host", port=1234, api_key="main_key", debug=False
    )
    mock_parse_args.return_value = mock_args
    test_exception = ValueError("Server init failed")
    mock_server_class.side_effect = test_exception
    mock_log_instance = MagicMock()
    mock_logger_class.return_value = mock_log_instance

    main()

    mock_parse_args.assert_called_once()
    mock_logger_class.assert_called_with("transcription_pipeline_manager.rest_interface", debug=False)
    mock_server_class.assert_called_once_with(
        host="main_host", port=1234, api_key="main_key", debug=False
    )
    mock_server_instance = mock_server_class.return_value
    mock_server_instance.start.assert_not_called()
    mock_log_instance.critical.assert_called_once_with(
        f"An unexpected error occurred in main: {test_exception}", exc_info=False
    )
    mock_print_exc.assert_called_once()
    mock_server_instance.shutdown.assert_not_called()
    mock_exit.assert_called_once_with(const.EXIT_FAILURE)
