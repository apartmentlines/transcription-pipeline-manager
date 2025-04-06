"""
Unit tests for the configuration loading and environment variable setting utilities.
"""

import argparse
import os
from collections.abc import Generator

import pytest
from pytest import MonkeyPatch

from transcription_pipeline_manager.config import (
    load_configuration,
    set_environment_variables,
)

# Constants for testing
TEST_API_KEY_ARG = "arg_api_key_123"
TEST_DOMAIN_ARG = "arg.example.com"
TEST_API_KEY_ENV = "env_api_key_456"
TEST_DOMAIN_ENV = "env.example.com"
ENV_API_KEY_NAME = "TRANSCRIPTION_API_KEY"
ENV_DOMAIN_NAME = "TRANSCRIPTION_DOMAIN"


@pytest.fixture
def mock_args() -> argparse.Namespace:
    """Fixture to create a mock argparse.Namespace object."""
    return argparse.Namespace(api_key=None, domain=None)


@pytest.fixture(autouse=True)
def clean_env_vars(monkeypatch: MonkeyPatch) -> Generator[None, None, None]:
    """
    Fixture to ensure relevant environment variables are cleaned before/after each test.
    Using autouse=True to apply it automatically to all tests in this module.
    """
    monkeypatch.delenv(ENV_API_KEY_NAME, raising=False)
    monkeypatch.delenv(ENV_DOMAIN_NAME, raising=False)
    yield
    # Teardown is implicitly handled by monkeypatch


# --- Tests for load_configuration ---


def test_load_config_from_args(mock_args: argparse.Namespace) -> None:
    """Verify config is loaded correctly when provided solely via args."""
    mock_args.api_key = TEST_API_KEY_ARG
    mock_args.domain = TEST_DOMAIN_ARG

    api_key, domain = load_configuration(mock_args)

    assert api_key == TEST_API_KEY_ARG
    assert domain == TEST_DOMAIN_ARG


def test_load_config_from_env(
    mock_args: argparse.Namespace, monkeypatch: MonkeyPatch
) -> None:
    """Verify config is loaded correctly when provided solely via environment variables."""
    monkeypatch.setenv(ENV_API_KEY_NAME, TEST_API_KEY_ENV)
    monkeypatch.setenv(ENV_DOMAIN_NAME, TEST_DOMAIN_ENV)

    api_key, domain = load_configuration(mock_args)

    assert api_key == TEST_API_KEY_ENV
    assert domain == TEST_DOMAIN_ENV


def test_load_config_args_override_env(
    mock_args: argparse.Namespace, monkeypatch: MonkeyPatch
) -> None:
    """Verify config from args takes precedence over environment variables."""
    mock_args.api_key = TEST_API_KEY_ARG
    mock_args.domain = TEST_DOMAIN_ARG
    monkeypatch.setenv(ENV_API_KEY_NAME, TEST_API_KEY_ENV)
    monkeypatch.setenv(ENV_DOMAIN_NAME, TEST_DOMAIN_ENV)

    api_key, domain = load_configuration(mock_args)

    assert api_key == TEST_API_KEY_ARG
    assert domain == TEST_DOMAIN_ARG


def test_load_config_mixed_sources_api_key_arg(
    mock_args: argparse.Namespace, monkeypatch: MonkeyPatch
) -> None:
    """Verify config loads correctly with API key from args, domain from env."""
    mock_args.api_key = TEST_API_KEY_ARG
    monkeypatch.setenv(ENV_DOMAIN_NAME, TEST_DOMAIN_ENV)

    api_key, domain = load_configuration(mock_args)

    assert api_key == TEST_API_KEY_ARG
    assert domain == TEST_DOMAIN_ENV


def test_load_config_mixed_sources_domain_arg(
    mock_args: argparse.Namespace, monkeypatch: MonkeyPatch
) -> None:
    """Verify config loads correctly with API key from env, domain from args."""
    monkeypatch.setenv(ENV_API_KEY_NAME, TEST_API_KEY_ENV)
    mock_args.domain = TEST_DOMAIN_ARG

    api_key, domain = load_configuration(mock_args)

    assert api_key == TEST_API_KEY_ENV
    assert domain == TEST_DOMAIN_ARG


def test_load_config_missing_api_key_raises_error(
    mock_args: argparse.Namespace, monkeypatch: MonkeyPatch
) -> None:
    """Verify ValueError is raised if API key is missing."""
    monkeypatch.setenv(ENV_DOMAIN_NAME, TEST_DOMAIN_ENV)

    with pytest.raises(
        ValueError,
        match="API key and domain must be provided",
    ):
        load_configuration(mock_args)


def test_load_config_missing_domain_raises_error(
    mock_args: argparse.Namespace, monkeypatch: MonkeyPatch
) -> None:
    """Verify ValueError is raised if domain is missing."""
    monkeypatch.setenv(ENV_API_KEY_NAME, TEST_API_KEY_ENV)

    with pytest.raises(
        ValueError,
        match="API key and domain must be provided",
    ):
        load_configuration(mock_args)


def test_load_config_missing_both_raises_error(mock_args: argparse.Namespace) -> None:
    """Verify ValueError is raised if both API key and domain are missing."""

    with pytest.raises(
        ValueError,
        match="API key and domain must be provided",
    ):
        load_configuration(mock_args)


# --- Tests for set_environment_variables ---


def test_set_env_vars_both(monkeypatch: MonkeyPatch) -> None:
    """Verify both environment variables are set correctly."""
    assert ENV_API_KEY_NAME not in os.environ
    assert ENV_DOMAIN_NAME not in os.environ

    set_environment_variables(TEST_API_KEY_ENV, TEST_DOMAIN_ENV)

    assert os.environ.get(ENV_API_KEY_NAME) == TEST_API_KEY_ENV
    assert os.environ.get(ENV_DOMAIN_NAME) == TEST_DOMAIN_ENV


def test_set_env_vars_only_api_key(monkeypatch: MonkeyPatch) -> None:
    """Verify only the API key environment variable is set."""
    assert ENV_API_KEY_NAME not in os.environ
    assert ENV_DOMAIN_NAME not in os.environ

    set_environment_variables(TEST_API_KEY_ENV, None)

    assert os.environ.get(ENV_API_KEY_NAME) == TEST_API_KEY_ENV
    assert ENV_DOMAIN_NAME not in os.environ


def test_set_env_vars_only_domain(monkeypatch: MonkeyPatch) -> None:
    """Verify only the domain environment variable is set."""
    assert ENV_API_KEY_NAME not in os.environ
    assert ENV_DOMAIN_NAME not in os.environ

    set_environment_variables(None, TEST_DOMAIN_ENV)

    assert ENV_API_KEY_NAME not in os.environ
    assert os.environ.get(ENV_DOMAIN_NAME) == TEST_DOMAIN_ENV


def test_set_env_vars_neither(monkeypatch: MonkeyPatch) -> None:
    """Verify no environment variables are set when both inputs are None."""
    assert ENV_API_KEY_NAME not in os.environ
    assert ENV_DOMAIN_NAME not in os.environ

    set_environment_variables(None, None)

    assert ENV_API_KEY_NAME not in os.environ
    assert ENV_DOMAIN_NAME not in os.environ
