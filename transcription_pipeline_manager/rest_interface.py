#!/usr/bin/env python3
"""
A simple Flask server to receive and log POST requests for transcription pipeline runs.

Listens on a specified port and prints the body of any incoming POST request.
If the Content-Type is application/json, it prints the values for each key in the JSON payload.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
import traceback
from typing import Any

from flask import Flask, Request, jsonify, request

from . import constants as const
from .logger import Logger


class Stats:
    """
    Thread-safe storage for application statistics. Internal to CallbackServer.
    """

    def __init__(self) -> None:
        """
        Initializes statistics and the lock.
        """
        self.pods_count: int = 0
        self.pipeline_last_run_time: int = 0
        self._lock: threading.Lock = threading.Lock()

    def get_pods_count(self) -> int:
        """
        Returns the current pods_count.

        :return: The current count of pods.
        :rtype: int
        """
        with self._lock:
            return self.pods_count

    def set_pods_count(self, count: int) -> None:
        """
        Sets the pods_count.

        :param count: The new count of pods.
        :type count: int
        """
        with self._lock:
            self.pods_count = count

    def get_pipeline_last_run_time(self) -> int:
        """
        Returns the timestamp of the last pipeline run notification.

        :return: The Unix timestamp of the last run.
        :rtype: int
        """
        with self._lock:
            return self.pipeline_last_run_time

    def set_pipeline_last_run_time(self, timestamp: int) -> None:
        """
        Sets the timestamp of the last pipeline run notification.

        :param timestamp: The Unix timestamp of the last run.
        :type timestamp: int
        """
        with self._lock:
            self.pipeline_last_run_time = timestamp


class CallbackServer:
    """
    Encapsulates the Flask application for handling callback requests.
    """

    def __init__(self, host: str, port: int, api_key: str | None = None, debug: bool = False) -> None:
        """
        Initializes the CallbackServer.

        :param host: The hostname to listen on.
        :type host: str
        :param port: The port to listen on.
        :type port: int
        :param api_key: The API key to access endpoints.
        :type api_key: str | None
        :param debug: Flag to enable debug logging.
        :type debug: bool
        """
        self.host: str = host
        self.port: int = port
        self.api_key: str | None = api_key or os.environ.get("TRANSCRIPTION_API_KEY")
        self.debug: bool = debug
        self.log: logging.Logger = Logger(self.__class__.__name__, debug=self.debug)
        self.app: Flask = Flask(__name__)
        self.stats: Stats = Stats()
        self.thread: threading.Thread | None = None
        self.stop_event: threading.Event = threading.Event()
        self._register_request_hooks()
        self.log.debug("CallbackServer initialized.")

    def _register_request_hooks(self) -> None:
        """Registers Flask request hooks."""
        self.app.before_request(self._check_api_key)
        self.log.debug("Registered before_request hooks.")

    def _check_api_key(self) -> tuple[Any, int] | None:
        """
        Checks for API key on incoming requests if configured.
        This function is registered with Flask's `before_request`.

        :return: None if access is allowed, or a (response, status_code) tuple if forbidden.
        :rtype: tuple[Any, int] | None
        """
        if self.api_key:
            provided_key = request.args.get("api_key")
            if not provided_key:
                self.log.warning(f"Forbidden: Missing API key for request to {request.path}")
                return jsonify({"error": "Forbidden", "message": "API key required"}), 403
            if provided_key != self.api_key:
                self.log.warning(f"Forbidden: Invalid API key provided for request to {request.path}")
                return jsonify({"error": "Forbidden", "message": "Invalid API key"}), 403
            self.log.debug(f"API key validated for request to {request.path}")
        else:
            self.log.debug(f"No API key configured, allowing request to {request.path}")
        return None

    def _register_routes(self) -> None:
        """
        Registers the Flask routes for the server.
        """
        self.app.add_url_rule(
            "/stats/pods-count",
            endpoint="get_pods_count",
            view_func=self.get_pods_count_handler,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/stats/pipeline-last-run-time",
            endpoint="get_pipeline_last_run_time",
            view_func=self.get_last_run_time_handler,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/logs",
            endpoint="handle_logs",
            view_func=self.handle_logs_handler,
            methods=["POST"],
        )
        self.log.debug("Registered routes.")

    def get_pods_count_handler(self) -> tuple[Any, int]:
        """
        Handles GET requests for the current pod count.

        :return: A tuple containing the JSON response and status code.
        :rtype: tuple[Any, int]
        """
        self.log.debug(f"Received GET request to: {request.path}")
        try:
            count = self.stats.get_pods_count()
            return jsonify({"pods_count": count}), 200
        except Exception as e:
            self.log.error(f"Error processing request to {request.path}: {e}", exc_info=self.debug)
            return jsonify({"error": "Internal Server Error"}), 500

    def get_last_run_time_handler(self) -> tuple[Any, int]:
        """
        Handles GET requests for the last pipeline run time.

        :return: A tuple containing the JSON response and status code.
        :rtype: tuple[Any, int]
        """
        self.log.debug(f"Received GET request to: {request.path}")
        try:
            timestamp = self.stats.get_pipeline_last_run_time()
            return jsonify({"pipeline_last_run_time": timestamp}), 200
        except Exception as e:
            self.log.error(f"Error processing request to {request.path}: {e}", exc_info=self.debug)
            return jsonify({"error": "Internal Server Error"}), 500

    def handle_logs_handler(self) -> tuple[str, int]:
        """
        Handles incoming POST requests to /logs, logs the body, and pretty-prints JSON.

        :return: A tuple containing the response text and status code.
        :rtype: tuple[str, int]
        """
        self.log.info(f"Received POST request to: {request.path}")
        req: Request = request
        content_type: str | None = req.headers.get("Content-Type")
        self.log.debug(f"Content-Type: {content_type}")
        try:
            data: bytes = req.get_data()
            if content_type == "application/json":
                try:
                    parsed_json: Any = json.loads(data)
                    self.log.debug("Received JSON payload")
                    # Mimic original behavior of printing key/value pairs
                    for key, value in parsed_json.items():
                        print(f"\n{key}:\n{value}")
                except json.JSONDecodeError:
                    self.log.warning(
                        "Content-Type is application/json but failed to parse body."
                    )
                    self.log.info(f"Raw body:\n{data.decode('utf-8', errors='replace')}")
            else:
                self.log.info(f"Received non-JSON body:\n{data.decode('utf-8', errors='replace')}")
            return "OK", 200
        except Exception as e:
            self.log.error(f"Error processing request to {request.path}: {e}", exc_info=self.debug)
            return "Internal Server Error", 500

    def start(self) -> None:
        """
        Starts the Flask server in a separate daemon thread.
        """
        self.log.info(f"Starting callback server on http://{self.host}:{self.port}")
        self._register_routes()

        def run_server():
            """Target function for the server thread."""
            try:
                # Note: Flask's dev server (app.run) doesn't easily support graceful shutdown via events.
                # Setting use_reloader=False is important when running in a thread.
                self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
            except Exception as e:
                self.log.critical(f"Callback server thread encountered an error: {e}", exc_info=self.debug)

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        self.log.debug("Callback server thread started.")

    def shutdown(self) -> None:
        """
        Signals the server thread to stop and waits for it to join.
        Note: Relies on the daemon nature for actual exit as app.run blocks.
        """
        self.log.info("Initiating callback server shutdown...")
        self.stop_event.set() # Signal intent, though app.run won't check it directly

        if self.thread is not None and self.thread.is_alive():
            # Give the thread a moment to potentially finish requests, though it won't exit cleanly
            # due to app.run blocking nature. The daemon=True is the primary exit mechanism.
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                self.log.info("Callback server thread did not join within timeout (expected with app.run).")
            else:
                self.log.info("Callback server thread joined.")
        else:
            self.log.info("Callback server thread was not running or already finished.")
        self.thread = None

    def update_pods_count(self, count: int) -> None:
        """
        Updates the pod count statistic.

        :param count: The new pod count.
        :type count: int
        """
        self.log.debug(f"Updating pods_count to {count}")
        self.stats.set_pods_count(count)

    def update_pipeline_last_run_time(self, timestamp: int) -> None:
        """
        Updates the pipeline last run time statistic.

        :param timestamp: The Unix timestamp of the last run.
        :type timestamp: int
        """
        self.log.debug(f"Updating pipeline_last_run_time to {timestamp}")
        self.stats.set_pipeline_last_run_time(timestamp)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: Parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Run a simple Flask server to catch POST callbacks."
    )
    parser.add_argument(
        "--host",
        type=str,
        default=const.DEFAULT_REST_HOST,
        help="Hostname to bind the server to (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=const.DEFAULT_REST_PORT,
        help="Port to bind the server to (default: %(default)s).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (also can be provided as TRANSCRIPTION_API_KEY environment variable)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging for the server script."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the callback server script when run directly.
    """
    args = parse_args()
    main_log = Logger(__name__, debug=args.debug)
    server = None
    exit_code = const.EXIT_SUCCESS

    try:
        server = CallbackServer(host=args.host, port=args.port, api_key=args.api_key, debug=args.debug)
        server.start()
        main_log.info("Callback server running in background thread. Press Ctrl+C to stop.")
        # Keep the main thread alive while the daemon server thread runs
        # Use a loop with sleep to allow KeyboardInterrupt
        while not server.stop_event.is_set():
            time.sleep(1) # Check every second
    except KeyboardInterrupt:
        main_log.info("Shutdown signal received (KeyboardInterrupt).")
    except Exception as e:
        main_log.critical(f"An unexpected error occurred in main: {e}", exc_info=args.debug)
        traceback.print_exc()
        exit_code = const.EXIT_FAILURE
    finally:
        if server:
            main_log.info("Initiating server shutdown...")
            server.shutdown()
        main_log.info("Server stopped.")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
