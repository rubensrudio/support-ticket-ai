"""Domain error type for the classification pipeline."""


class PipelineError(Exception):
    """Expected pipeline failure with a message safe to show to the user."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message

    def __str__(self) -> str:
        return self.message
