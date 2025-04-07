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
    NGROK_DOMAIN,
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
    POD_URL_TEMPLATE,
    # State constants
    STATE_STARTING_CYCLE,
    STATE_ATTEMPTING_POD_START,
    STATE_WAITING_FOR_IDLE,
    STATE_ATTEMPTING_PIPELINE_RUN,
    STATE_UPDATING_COUNTS,
    STATE_WAITING_AFTER_FAILURE,
    VALID_STATES,
)


class TranscriptionPipelineManager:
    """
    Manages the lifecycle of a transcription pipeline running on a RunPod pod.

    This class orchestrates starting a pod, waiting for it to become idle,
    triggering a processing run, monitoring its status (via REST interface updates),
    and handling cleanup, all within a defined hourly cycle.
    """
    def __init__(
        self,
        api_key: str,
        domain: str,
        limit: int = DEFAULT_TRANSCRIPTION_LIMIT,
        processing_limit: int = DEFAULT_TRANSCRIPTION_PROCESSING_LIMIT,
        ngrok: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Initializes the TranscriptionPipelineManager.

        Sets up configuration, logging, REST interface, RunPod managers,
        and the shutdown event mechanism.

        :param api_key: The API key for accessing protected endpoints and services.
        :type api_key: str
        :param domain: The base domain used for constructing service URLs.
        :type domain: str
        :param limit: The maximum number of items to process in a pipeline run.
        :type limit: int
        :param processing_limit: The concurrency limit for processing within the pipeline.
        :type processing_limit: int
        :param ngrok: Flag to enable ngrok tunneling.
        :type ngrok: bool
        :param debug: Flag to enable debug logging throughout the manager and its components.
        :type debug: bool
        """
        self.api_key: str = api_key
        self.domain: str = domain
        self.limit: int = limit
        self.processing_limit: int = processing_limit
        self.ngrok: bool = ngrok
        self.debug: bool = debug

        self.log: logging.Logger = Logger(self.__class__.__name__, debug=self.debug)
        self.log.info("Initializing Transcription Pipeline Manager...")

        self.callback_url_subdomain: str = f"www.{self.domain}"
        self.transcription_subdomain: str = f"my.{self.domain}"
        self.logs_callback_url: str = self._build_logs_callback_url()
        self.log.debug(f"Callback URL Subdomain: {self.callback_url_subdomain}")
        self.log.debug(f"Transcription Subdomain: {self.transcription_subdomain}")
        self.log.debug(f"Logs Callback URL: {self.logs_callback_url}")

        base_path = Path(__file__).parent.parent
        self.runpod_config_path: Path = base_path / RUNPOD_CONFIG_DIR / RUNPOD_CONFIG_FILENAME
        self.log.debug(f"RunPod config path: {self.runpod_config_path}")

        self.rest_interface: RestInterface = RestInterface(
            host=DEFAULT_REST_HOST,
            port=DEFAULT_REST_PORT,
            api_key=self.api_key,
            debug=self.debug
        )
        self.log.debug("REST Interface initialized.")

        self.runpod_start_manager: RunpodSingletonManager = RunpodSingletonManager(
            config_path=self.runpod_config_path,
            debug=self.debug,
        )
        self.log.debug("RunPod Start Manager initialized.")
        self.runpod_terminate_manager: RunpodSingletonManager = RunpodSingletonManager(
            config_path=self.runpod_config_path,
            debug=self.debug,
            terminate=True
        )
        self.log.debug("RunPod Terminate Manager initialized.")

        self.shutdown_event: threading.Event = threading.Event()
        self.log.debug("Shutdown event initialized.")
        self.log.debug("Transcription Pipeline Manager initialization complete.")

    def _build_logs_callback_url(self) -> str:
        """Constructs the URL for the REST interface's log endpoint."""
        if self.ngrok:
            return f"{NGROK_DOMAIN}/logs?api_key={self.api_key}"
        else:
            return f"https://{self.callback_url_subdomain}/api/transcription/logs?api_key={self.api_key}"

    def get_current_time(self) -> float:
        """Returns the current time in seconds since the Unix epoch."""
        return time.time()

    def sleep(self, seconds: float) -> None:
        """Sleeps for the specified number of seconds."""
        time.sleep(seconds)

    def run(self) -> None:
        """
        The main execution method that starts the manager's control loop.

        Initializes signal handling, starts the REST interface, and runs the
        state machine loop until a shutdown signal is received.
        """
        self.log.debug("Starting transcription pipeline manager run loop...")
        self._setup_signal_handlers()
        self.rest_interface.start()

        current_state: str = STATE_STARTING_CYCLE
        cycle_start_time: float = 0.0
        pod_id: str = ""
        pod_url: str = ""
        last_count_update_time: float = 0.0
        last_idle_check_time: float = 0.0

        try:
            while not self.shutdown_event.is_set():
                now = self.get_current_time()
                elapsed_cycle_time = now - cycle_start_time if cycle_start_time > 0 else 0

                # --- Time-Based Cycle Reset ---
                if current_state != STATE_STARTING_CYCLE and elapsed_cycle_time >= CYCLE_DURATION:
                    self.log.info(f"Hourly cycle duration ({CYCLE_DURATION}s) reached. Resetting cycle.")
                    current_state = STATE_STARTING_CYCLE
                    # Pod termination is handled by pod itself or next start cycle

                # --- State Dispatch ---
                if current_state == STATE_STARTING_CYCLE:
                    current_state, cycle_start_time, pod_id, pod_url, last_count_update_time, last_idle_check_time = self._handle_starting_cycle(now)

                if current_state == STATE_ATTEMPTING_POD_START:
                    current_state, pod_id, pod_url = self._handle_attempting_pod_start()

                if current_state == STATE_WAITING_FOR_IDLE:
                    current_state, last_idle_check_time = self._handle_waiting_for_idle(
                        now, elapsed_cycle_time, last_idle_check_time, pod_id, pod_url
                    )

                if current_state == STATE_ATTEMPTING_PIPELINE_RUN:
                    current_state = self._handle_attempting_pipeline_run(pod_id, pod_url)

                if current_state == STATE_UPDATING_COUNTS:
                    current_state, last_count_update_time = self._handle_updating_counts(
                        now, last_count_update_time
                    )

                if current_state == STATE_WAITING_AFTER_FAILURE:
                    current_state = self._handle_waiting_after_failure(elapsed_cycle_time)

                if current_state not in VALID_STATES:
                    raise RuntimeError(f"Encountered unknown or invalid state: {current_state}. Aborting.")

                # --- End of Loop ---
                self.sleep(MAIN_LOOP_SLEEP)

        except KeyboardInterrupt:
             self.log.info("KeyboardInterrupt received, initiating shutdown.")
             self.shutdown_event.set()
        except Exception as e:
             self.log.critical(f"Unhandled exception in main loop: {e}", exc_info=self.debug)
             self.shutdown_event.set()
        finally:
            self.log.info("Run loop finished or interrupted.")
            self._shutdown()
            self.log.info("Transcription pipeline manager shut down gracefully.")

    # --- State Handler Methods ---

    def _handle_starting_cycle(self, now: float) -> tuple[str, float, str, str, float, float]:
        """Handles logic for the STARTING_CYCLE state."""
        self.log.debug("Starting new hourly cycle.")
        next_state = STATE_ATTEMPTING_POD_START
        cycle_start_time = now
        pod_id = ""
        pod_url = ""
        last_count_update_time = 0.0
        last_idle_check_time = 0.0
        return next_state, cycle_start_time, pod_id, pod_url, last_count_update_time, last_idle_check_time

    def _handle_attempting_pod_start(self) -> tuple[str, str, str]:
        """Handles logic for the ATTEMPTING_POD_START state."""
        self.log.debug("Attempting to ensure RunPod pod is running...")
        next_state = STATE_WAITING_AFTER_FAILURE
        pod_id = ""
        pod_url = ""
        try:
            pod_id_return_value = self.runpod_start_manager.run()
            if pod_id_return_value:
                pod_id = str(pod_id_return_value)
                pod_url = POD_URL_TEMPLATE % pod_id
                self.log.info(f"Pod {pod_id} confirmed running/started. URL: {pod_url}")
                next_state = STATE_WAITING_FOR_IDLE
            else:
                self.log.error("Failed to start or confirm pod is running.")
                self._terminate_pods()
        except Exception as e:
            self.log.error(f"Error during pod start attempt: {e}", exc_info=self.debug)
            self._terminate_pods()
        return next_state, pod_id, pod_url

    def _handle_waiting_for_idle(self, now: float, elapsed_cycle_time: float, last_idle_check_time: float, pod_id: str, pod_url: str) -> tuple[str, float]:
        """Handles logic for the WAITING_FOR_IDLE state."""
        next_state = STATE_WAITING_FOR_IDLE
        updated_last_idle_check_time = last_idle_check_time
        if elapsed_cycle_time >= STATUS_CHECK_TIMEOUT:
            self.log.warning(f"Timeout ({STATUS_CHECK_TIMEOUT}s) waiting for pod {pod_id} to become idle. Terminating.")
            self._terminate_pods()
            next_state = STATE_WAITING_AFTER_FAILURE
        elif now - last_idle_check_time >= POD_STATUS_CHECK_INTERVAL:
            self.log.debug(f"Checking if pod {pod_id} is idle at {pod_url}...")
            updated_last_idle_check_time = now
            is_idle = self._check_pod_idle_status(pod_url)
            if is_idle:
                self.log.info(f"Pod {pod_id} at {pod_url} is now ready.")
                next_state = STATE_ATTEMPTING_PIPELINE_RUN
            else:
                self.log.debug(f"Pod {pod_id} not ready yet. Waiting...")
        return next_state, updated_last_idle_check_time

    def _handle_attempting_pipeline_run(self, pod_id: str, pod_url: str) -> str:
        """Handles logic for the ATTEMPTING_PIPELINE_RUN state."""
        self.log.debug(f"Attempting to trigger pipeline run on pod {pod_id}...")
        run_triggered = self._trigger_pipeline_run(pod_url)
        if run_triggered:
            self.log.info(f"Pipeline run successfully triggered on pod {pod_id}.")
            return STATE_UPDATING_COUNTS
        else:
            self.log.error(f"Failed to trigger pipeline run on pod {pod_id}. Terminating.")
            self._terminate_pods()
            return STATE_WAITING_AFTER_FAILURE

    def _handle_updating_counts(self, now: float, last_count_update_time: float) -> tuple[str, float]:
        """Handles logic for the UPDATING_COUNTS state."""
        next_state = STATE_UPDATING_COUNTS
        updated_last_count_update_time = last_count_update_time
        if now - last_count_update_time >= COUNT_UPDATE_INTERVAL:
            self.log.debug("Updating pod counts...")
            counts_updated = self._update_pod_counts()
            if counts_updated:
                updated_last_count_update_time = now
        return next_state, updated_last_count_update_time

    def _handle_waiting_after_failure(self, elapsed_cycle_time: float) -> str:
        """Handles logic for the WAITING_AFTER_FAILURE state."""
        remaining_time = max(0, CYCLE_DURATION - elapsed_cycle_time)
        self.log.warning(f"In failure state. Waiting for next cycle. Time remaining: {remaining_time:.0f}s")
        return STATE_WAITING_AFTER_FAILURE

    # --- Core Helper Methods ---

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
                self.log.error(f"Failed to decode JSON response from {status_url}.", exc_info=self.debug)
                return False
        except requests.exceptions.HTTPError as e:
            self.log.debug(f"Failed to get pod status from {status_url}. Status: {e.response.status_code} {e.response.reason}")
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
            self.log.error(f"Failed to decode JSON response from {run_url}: {e}.", exc_info=self.debug)
            return False
        if "message" in response_data:
            self.log.info(f"Pipeline run triggered successfully: {response_data['message']}")
            self.rest_interface.update_pipeline_last_run_time(int(time.time())) # Update REST interface stats on successful trigger
            return True
        elif "error" in response_data:
            self.log.error(f"Pipeline run trigger failed at {run_url}: {response_data['error']}")
            return False
        else:
            self.log.error(f"Unexpected JSON response format from {run_url}: {response_data}")
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
        self.log.debug(f"Payload: {payload}")
        try:
            response = requests.post(run_url, json=payload, timeout=POD_REQUEST_TIMEOUT)
            response.raise_for_status()
            return self._process_trigger_pipeline_run_response(run_url, response)
        except requests.exceptions.HTTPError as e:
            self.log.error(f"Failed to trigger pipeline run at {run_url}. Status: {e.response.status_code} {e.response.reason}")
            return False
        except Exception as e:
            self.log.error(f"Error triggering pipeline run at {run_url}: {e}", exc_info=self.debug)
            return False

    def _terminate_pods(self) -> None:
        """
        Attempts to stop and terminate any existing pods matching the configuration
        using the dedicated RunpodSingletonManager instance.
        """
        self.log.debug("Attempting to terminate any existing pods...")
        try:
            # The terminate manager is configured with stop=True, terminate=True
            result = self.runpod_terminate_manager.run()
            if result:
                self.log.info("Pod termination/stop process completed.")
            else:
                # runpod-singleton returns None on failure/no action needed
                self.log.error("Pod termination/stop process did not complete successfully.")
        except Exception as e:
            self.log.error(f"An error occurred during pod termination/stop: {e}", exc_info=self.debug)

    def _update_pod_counts(self) -> bool:
        """
        Retrieves the current total and running pod counts using the
        RunpodSingletonManager and updates the RestInterface statistics.

        :return: True if pod counts were successfully updated, False otherwise.
        :rtype: bool
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
                return True
            else:
                 self.log.warning("Did not receive valid pod counts from RunpodSingletonManager.")
        except Exception as e:
            self.log.error(f"Failed to retrieve pod counts: {e}", exc_info=self.debug)
        return False

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

    def _shutdown(self) -> None:
        """
        Performs graceful shutdown of manager components, primarily the RestInterface.
        """
        self.log.info("Shutting down REST interface...")
        if self.rest_interface:
            try:
                self.rest_interface.shutdown()
            except Exception as e:
                self.log.error(f"Error shutting down REST interface: {e}", exc_info=self.debug)
        else:
            self.log.debug("REST interface was not initialized, nothing to shut down.")
        self.log.debug("REST interface shutdown complete.")


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
    parser.add_argument("--ngrok", action="store_true", help="Use ngrok to expose the REST interface")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    try:
        api_key, domain = load_configuration(args)
        pipeline = TranscriptionPipelineManager(
            api_key=api_key,
            domain=domain,
            limit=args.limit,
            processing_limit=args.processing_limit,
            ngrok=args.ngrok,
            debug=args.debug,
        )
        pipeline.run()
    except Exception as e:
        logging.getLogger(__name__).critical(f"Unhandled exception during manager execution: {e}", exc_info=args.debug)
        fail_hard("Transcription Pipeline Manager failed due to an unhandled exception.")


if __name__ == "__main__":
    main()
