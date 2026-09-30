import argparse

import pytest

from ticket_classifier import cli
from ticket_classifier.errors import PipelineError


def test_pipeline_error_keeps_message() -> None:
    exc = PipelineError("boom")

    assert exc.message == "boom"
    assert str(exc) == "boom"
    assert isinstance(exc, Exception)


def _parser_with_handler(handler: cli.Handler) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ticket-classifier")
    subparsers = parser.add_subparsers(dest="command")
    fail_parser = subparsers.add_parser("fail")
    fail_parser.set_defaults(handler=handler)
    return parser


def test_main_returns_one_and_prints_message_on_pipeline_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_handler(args: argparse.Namespace) -> int:
        raise PipelineError("boom")

    monkeypatch.setattr(cli, "build_parser", lambda: _parser_with_handler(failing_handler))

    exit_code = cli.main(["fail"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "boom" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_main_returns_handler_exit_code_without_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "build_parser", lambda: _parser_with_handler(lambda args: 0))

    assert cli.main(["fail"]) == 0
