#!/usr/bin/env python3

import argparse
import logging
import signal
import threading
import time
import json
from pathlib import Path
from copy import deepcopy
from typing import Any

import requests
from requests.exceptions import RequestException
from runpod_singleton import RunpodSingletonManager

from transcription_pipeline_manager.rest_interface import RestInterface
from transcription_pipeline_manager.utils import (
    fail_hard,
    positive_int,
)
from transcription_pipeline_manager.logger import Logger
from transcription_pipeline_manager.config import load_configuration, set_environment_variables
from transcription_pipeline_manager.constants import (
    DEFAULT_REST_HOST,
    DEFAULT_REST_PORT,
    DEFAULT_TRANSCRIPTION_LIMIT,
    DEFAULT_TRANSCRIPTION_PROCESSING_LIMIT,
    # Manager specific constants
    CYCLE_DURATION,
    STATUS_CHECK_TIMEOUT,
    COUNT_UPDATE_INTERVAL,
    MAIN_LOOP_SLEEP,
    POD_STATUS_CHECK_INTERVAL,
    POD_REQUEST_TIMEOUT,
    RUNPOD_CONFIG_FILENAME,
    RUNPOD_CONFIG_DIR,
)


class TranscriptionPipelineManager:
    def __init__(
        self,
        api_key: str | None = None,
        domain: str | None = None,
        limit: int | None = DEFAULT_TRANSCRIPTION_LIMIT,
        processing_limit: int | None = DEFAULT_TRANSCRIPTION_PROCESSING_LIMIT,
        debug: bool = False,
    ) -> None:
        self.log: logging.Logger = Logger(self.__class__.__name__, debug=debug)
        self.api_key: str | None = api_key
        self.domain: str | None = domain
        self.limit: int | None = limit
        self.processing_limit: int | None = processing_limit
        self.callback_url_subdomain: str = f"www.{self.domain}"
        self.transcription_subdomain: str = f"my.{self.domain}"
        self.logs_callback_url: str = self.build_logs_callback_url()
        self.debug: bool = debug
        # Initialize dependencies (will be properly instantiated later if needed)
        self.rest_interface: RestInterface | None = None
        self.runpod_start_manager: RunpodSingletonManager | None = None
        self.runpod_terminate_manager: RunpodSingletonManager | None = None
        self.shutdown_event: threading.Event = threading.Event()

    def build_logs_callback_url(self) -> str:
        return f"https://{self.callback_url_subdomain}/api/transcription/logs?api_key={self.api_key}"

    def setup_configuration(self) -> None:
        self.log.debug("Setting up configuration")
        if not self.api_key or not self.domain:
            fail_hard("API key and domain must be provided")
        set_environment_variables(self.api_key, self.domain)
        self.log.info("Configuration loaded successfully")

    def run(self) -> None:
        self.log.info("Starting transcription pipeline manager")
        self.setup_configuration()
        # TODO: Implement the main loop in run()

    # --- Helper Methods ---

    def _check_pod_idle_status(self, pod_url: str) -> bool:
        """
        Checks if the RunPod pod is reporting an 'idle' status via its /status endpoint.

        :param pod_url: The base URL of the pod
        :type pod_url: str
        :return: True if the pod status is 'idle', False otherwise.
        :rtype: bool
        """
        status_url = f"{pod_url}/status"
        self.log.debug(f"Checking pod status at {status_url}")
        try:
            response = requests.get(status_url, timeout=POD_REQUEST_TIMEOUT)
            response.raise_for_status()
            try:
                status_data = response.json()
                if isinstance(status_data, dict) and status_data.get("status") == "idle":
                    self.log.info(f"Pod at {pod_url} is ready.")
                    return True
                else:
                    status_value = status_data.get('status', 'N/A') if isinstance(status_data, dict) else 'Invalid Format'
                    self.log.debug(f"Pod at {pod_url} status: {status_value}")
                    return False
            except json.JSONDecodeError:
                self.log.warning(f"Failed to decode JSON response from {status_url}.", exc_info=self.debug)
                return False
        except requests.exceptions.HTTPError as e:
            self.log.warning(f"Failed to get pod status from {status_url}. Status: {e.response.status_code} {e.response.reason}")
            return False
        except RequestException as e:
            self.log.error(f"Error requesting pod status from {status_url}: {e}", exc_info=self.debug)
            return False

    def _process_trigger_pipeline_run_response(self, run_url: str, response: requests.Response) -> bool:
        """
        Processes the response from a pipeline run trigger request.

        :param run_url: The URL of the run endpoint
        :type run_url: str
        :param response: The response object from the request
        :type response: requests.Response
        :return: True if the pipeline run was successfully triggered, False otherwise.
        :rtype: bool
        """
        try:
            response_data = response.json()
        except Exception as e:
            self.log.warning(f"Failed to decode JSON response from {run_url}: {e}.", exc_info=self.debug)
            return False
        if "message" in response_data:
            self.log.info(f"Pipeline run triggered successfully: {response_data['message']}")
            self.rest_interface.update_pipeline_last_run_time(int(time.time()))
            return True
        elif "error" in response_data:
            self.log.error(f"Pipeline run trigger failed at {run_url}: {response_data['error']}")
            return False
        else:
            self.log.warning(f"Unexpected JSON response format from {run_url}: {response_data}")
            return False

    def _trigger_pipeline_run(self, pod_url: str) -> bool:
        """
        Triggers the transcription pipeline run via a POST request to the pod's /run endpoint.

        :param pod_url: The base URL of the pod
        :type pod_url: str
        :return: True if the pipeline run was successfully triggered, False otherwise.
        :rtype: bool
        """
        run_url = f"{pod_url}/run"
        payload = {
            "api_key": self.api_key,
            "domain": self.transcription_subdomain,
            "limit": self.limit,
            "processing_limit": self.processing_limit,
            "callback_url": self.logs_callback_url,
        }
        self.log.info(f"Triggering pipeline run at {run_url}")
        self.log.debug(f"Payload: {payload}") # Log sensitive data only in debug
        try:
            response = requests.post(run_url, json=payload, timeout=POD_REQUEST_TIMEOUT)
            response.raise_for_status()
            return self._process_trigger_pipeline_run_response(run_url, response)
        except requests.exceptions.HTTPError as e:
            self.log.warning(f"Failed to trigger pipeline run at {run_url}. Status: {e.response.status_code} {e.response.reason}")
            return False
        except Exception as e:
            self.log.error(f"Error triggering pipeline run at {run_url}: {e}", exc_info=self.debug)
            return False

    def _terminate_pods(self) -> None:
        """
        Attempts to stop and terminate any existing pods matching the configuration
        using the dedicated RunpodSingletonManager instance.
        """
        self.log.info("Attempting to terminate any existing pods...")
        try:
            result = self.runpod_terminate_manager.run()
            if result:
                self.log.info("Pod termination/stop process completed.")
            else:
                self.log.warning("Pod termination/stop process did not complete successfully.")
        except Exception as e:
            self.log.error(f"An error occurred during pod termination/stop: {e}", exc_info=self.debug)

    def _update_pod_counts(self) -> None:
        """
        Retrieves the current total and running pod counts using the
        RunpodSingletonManager and updates the RestInterface statistics.
        """
        self.log.debug("Attempting to update pod counts...")
        try:
            counts = self.runpod_start_manager.count_pods()
            if isinstance(counts, dict) and 'total' in counts and 'running' in counts:
                total = counts['total']
                running = counts['running']
                self.rest_interface.update_pods_total(total)
                self.rest_interface.update_pods_running(running)
                self.log.info(f"Successfully updated pod counts: Total={total}, Running={running}")
            else:
                 self.log.warning("Did not receive valid pod counts from RunpodSingletonManager.")
        except Exception as e:
            self.log.error(f"Failed to retrieve pod counts: {e}", exc_info=self.debug)

    def _setup_signal_handlers(self) -> None:
        """
        Registers signal handlers for graceful shutdown (SIGINT, SIGTERM).
        """
        self.log.debug("Registering signal handlers for SIGINT and SIGTERM.")
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)

    def _handle_shutdown_signal(self, signum: int, _frame: Any) -> None:
        """
        Signal handler function to initiate graceful shutdown.

        Sets the shutdown_event when SIGINT or SIGTERM is received.

        :param signum: The signal number received.
        :type signum: int
        :param frame: The current stack frame (unused).
        :type frame: Any
        """
        signal_name = signal.Signals(signum).name
        self.log.info(f"Received signal {signal_name}. Initiating graceful shutdown...")
        self.shutdown_event.set()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the processing pipeline manager.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (also can be provided as TRANSCRIPTION_API_KEY environment variable)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Transcription domain used for REST operations (also can be provided as TRANSCRIPTION_DOMAIN environment variable)",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=DEFAULT_TRANSCRIPTION_LIMIT,
        help="Only transcribe this many files, default unlimited",
    )
    parser.add_argument(
        "--processing-limit",
        type=positive_int,
        default=DEFAULT_TRANSCRIPTION_PROCESSING_LIMIT,
        help="Maximum concurrent processing threads, default %(default)s",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    try:
        api_key, domain = load_configuration(args)
    except ValueError as e:
        fail_hard(str(e))
        return

    pipeline = TranscriptionPipelineManager(
        api_key=api_key,
        domain=domain,
        limit=args.limit,
        processing_limit=args.processing_limit,
        debug=args.debug,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
