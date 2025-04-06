import argparse
import sys
import pytest
from unittest.mock import patch
from transcription_pipeline_manager.utils import (
    positive_int,
    fail_hard,
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
