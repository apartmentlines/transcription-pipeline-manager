#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from copy import deepcopy

from transcription_pipeline_manager.utils import (
    fail_hard,
    positive_int,
)
from transcription_pipeline_manager.logger import Logger
from transcription_pipeline_manager.config import load_configuration, set_environment_variables
from transcription_pipeline_manager.constants import (
    DEFAULT_REST_HOST,
    DEFAULT_REST_PORT,
)


class TranscriptionPipelineManager:
    def __init__(
        self,
        api_key: str | None = None,
        domain: str | None = None,
        debug: bool = False,
    ) -> None:
        self.log: logging.Logger = Logger(self.__class__.__name__, debug=debug)
        self.api_key: str = api_key
        self.domain: str = domain
        self.debug: bool = debug

    def build_retrieve_callback_url(self) -> str:
        return f"https://www.{self.domain}/api/transcription/batch-complete"

    def setup_configuration(self) -> None:
        self.log.debug("Setting up configuration")
        if not self.api_key or not self.domain:
            fail_hard("API key and domain must be provided")
        set_environment_variables(self.api_key, self.domain)
        self.log.info("Configuration loaded successfully")

    def run(self) -> None:
        self.log.info("Starting transcription pipeline manager")
        self.setup_configuration()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the processing pipeline.")
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (also can be provided as TRANSCRIPTION_API_KEY environment variable)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Transcription domain used for REST operations (also can be provided as TRANSCRIPTION_DOMAIN environment variable)",
    )
    parser.add_argument(
        "--simulate-downloads",
        action="store_true",
        help="Simulate downloads instead of performing actual downloads",
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
        debug=args.debug,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
