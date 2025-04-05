#!/usr/bin/env python3
"""
A simple Flask server to receive and log POST requests for testing callbacks.

Listens on a specified port and prints the body of any incoming POST request.
If the Content-Type is application/json, it pretty-prints the JSON payload.
"""

import argparse
import json
import logging
import sys
import traceback
from typing import Any

from flask import Flask, Request, request

from . import constants as const
from .logger import Logger


class CallbackServer:
    """
    Encapsulates the Flask application for handling callback requests.
    """

    def __init__(self, host: str, port: int, debug: bool = False) -> None:
        """
        Initializes the CallbackServer.

        :param host: The hostname to listen on.
        :type host: str
        :param port: The port to listen on.
        :type port: int
        :param debug: Flag to enable debug logging.
        :type debug: bool
        """
        self.host: str = host
        self.port: int = port
        self.debug: bool = debug
        self.log: logging.Logger = Logger(self.__class__.__name__, debug=self.debug)
        self.app: Flask = Flask(__name__)
        self._register_routes()
        self.log.debug("CallbackServer initialized.")

    def _register_routes(self) -> None:
        """
        Registers the Flask routes for the server.
        """
        self.app.add_url_rule(
            "/<path:path>",
            endpoint="catch_all_post",
            view_func=self.handle_post,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/",
            endpoint="catch_root_post",
            view_func=self.handle_post,
            methods=["POST"],
            defaults={"path": ""},
        )
        self.log.debug("Registered POST route for all paths.")

    def handle_post(self, path: str) -> tuple[str, int]:
        """
        Handles incoming POST requests, logs the body, and pretty-prints JSON.

        :param path: The path requested (captured from the URL rule).
        :type path: str
        :return: A tuple containing the response text and status code.
        :rtype: tuple[str, int]
        """
        full_path: str = f"/{path}"
        self.log.info(f"Received POST request to: {full_path}")
        req: Request = request
        content_type: str | None = req.headers.get("Content-Type")
        self.log.debug(f"Content-Type: {content_type}")
        try:
            data: bytes = req.get_data()
            if content_type == "application/json":
                try:
                    parsed_json: Any = json.loads(data)
                    pretty_json: str = json.dumps(parsed_json, indent=2)
                    self.log.info(f"Received JSON payload:\n{pretty_json}")
                except json.JSONDecodeError:
                    self.log.warning(
                        "Content-Type is application/json but failed to parse body."
                    )
                    self.log.info(f"Raw body:\n{data.decode('utf-8', errors='replace')}")
            else:
                self.log.info(f"Received non-JSON body:\n{data.decode('utf-8', errors='replace')}")
            return "OK", 200
        except Exception as e:
            self.log.error(f"Error processing request to {full_path}: {e}", exc_info=self.debug)
            return "Internal Server Error", 500

    def run(self) -> None:
        """
        Starts the Flask development server.
        """
        self.log.info(f"Starting callback test server on http://{self.host}:{self.port}")
        try:
            self.app.run(host=self.host, port=self.port, debug=False)
        except Exception as e:
            self.log.critical(f"Failed to start server: {e}", exc_info=self.debug)
            sys.exit(const.EXIT_FAILURE)


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
        default=const.DEFAULT_TEST_CALLBACK_HOST,
        help="Hostname to bind the server to (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=const.DEFAULT_TEST_CALLBACK_PORT,
        help="Port to bind the server to (default: %(default)s).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging for the server script."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the test callback server script.
    """
    args = parse_args()

    try:
        server = CallbackServer(host=args.host, port=args.port, debug=args.debug)
        server.run()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(const.EXIT_FAILURE)


if __name__ == "__main__":
    main()
