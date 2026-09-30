import argparse

import pytest

from ticket_classifier.cli import build_parser, main


def test_build_parser_returns_argument_parser() -> None:
    parser = build_parser()

    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "ticket-classifier"


def test_main_help_exits_with_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "ticket-classifier" in capsys.readouterr().out


def test_main_without_command_returns_non_zero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])

    assert exit_code != 0
    assert "usage" in capsys.readouterr().err


def test_ops03_offline_env_is_forced_by_conftest() -> None:
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
