"""Resolve user-writable paths, including paths for the portable Windows build."""

import sys
import tempfile
from pathlib import Path

from platformdirs import user_data_dir, user_log_dir


PORTABLE_MARKER = "scoresight.portable"


def is_portable() -> bool:
    """Return whether this frozen app was packaged as a portable build."""
    if not getattr(sys, "frozen", False):
        return False
    executable_dir = Path(sys.executable).resolve().parent
    return (executable_dir / PORTABLE_MARKER).is_file()


def get_user_data_dir() -> str:
    """Use a folder next to the executable for portable-build settings."""
    if is_portable():
        data_dir = Path(sys.executable).resolve().parent / "data"
        try:
            data_dir.mkdir(exist_ok=True)
            with tempfile.TemporaryFile(dir=data_dir):
                pass
            return str(data_dir)
        except OSError:
            # A read-only USB drive can still run the app, with settings in the
            # normal per-user location instead.
            pass
    return user_data_dir("scoresight")


def get_user_log_dir() -> str:
    """Keep portable logs with portable settings; otherwise use platformdirs."""
    if is_portable():
        return str(Path(get_user_data_dir()) / "logs")
    return user_log_dir("scoresight")
