"""
TP5 - Step 1 Ex 1.2: Reusable logging utility (rich-based)
Import this in any TP5 script: from tp5_log import log, log_success, log_error
"""

from rich.console import Console
from rich.theme import Theme

_theme = Theme({
    "info":    "bold cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error":   "bold red",
    "step":    "bold magenta",
})

_console = Console(theme=_theme)


def log(message: str) -> None:
    """Print a standard info message."""
    _console.print(f"[info]ℹ[/info]  {message}")


def log_success(message: str) -> None:
    """Print a success message."""
    _console.print(f"[success]✔[/success]  {message}")


def log_error(message: str) -> None:
    """Print an error message."""
    _console.print(f"[error]✘[/error]  {message}")


def log_step(title: str) -> None:
    """Print a section header."""
    _console.rule(f"[step]{title}[/step]")


def log_warning(message: str) -> None:
    """Print a warning message."""
    _console.print(f"[warning]⚠[/warning]  {message}")


if __name__ == "__main__":
    log_step("TP5 Logger — self-test")
    log("This is an info message")
    log_success("This is a success message")
    log_warning("This is a warning message")
    log_error("This is an error message")
