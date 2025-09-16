"""Embedded toy datasets used by the sandbox."""
from importlib.resources import files
from pathlib import Path
from typing import Iterator

_PACKAGE = __package__


def data_path(name: str) -> Path:
    """Return the filesystem path to a packaged dataset file."""

    resource = files(_PACKAGE).joinpath(name)
    if not resource.is_file():
        raise FileNotFoundError(f"Dataset '{name}' not found in {__package__}")
    return Path(resource)


def list_datasets() -> Iterator[str]:
    """Yield dataset filenames bundled with the package."""

    resource = files(_PACKAGE)
    for entry in resource.iterdir():
        if entry.is_file() and entry.name.lower().endswith(('.csv', '.npz')):
            yield entry.name


__all__ = ["data_path", "list_datasets"]
