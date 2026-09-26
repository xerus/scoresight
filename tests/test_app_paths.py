import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app_paths


class AppPathTests(unittest.TestCase):
    def test_portable_build_uses_data_and_logs_next_to_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "scoresight.exe"
            executable.touch()
            (root / app_paths.PORTABLE_MARKER).touch()

            with (
                patch.object(app_paths.sys, "frozen", True, create=True),
                patch.object(app_paths.sys, "executable", str(executable)),
            ):
                self.assertEqual(app_paths.get_user_data_dir(), str(root / "data"))
                self.assertEqual(
                    app_paths.get_user_log_dir(), str(root / "data" / "logs")
                )

    def test_read_only_portable_folder_falls_back_to_user_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "scoresight.exe"
            executable.touch()
            (root / app_paths.PORTABLE_MARKER).touch()

            with (
                patch.object(app_paths.sys, "frozen", True, create=True),
                patch.object(app_paths.sys, "executable", str(executable)),
                patch.object(
                    app_paths.tempfile,
                    "TemporaryFile",
                    side_effect=PermissionError("read-only drive"),
                ),
                patch.object(app_paths, "user_data_dir", return_value="user-data"),
            ):
                self.assertEqual(app_paths.get_user_data_dir(), "user-data")

    def test_source_build_uses_platformdirs(self):
        with (
            patch.object(app_paths.sys, "frozen", False, create=True),
            patch.object(app_paths, "user_data_dir", return_value="user-data"),
        ):
            self.assertEqual(app_paths.get_user_data_dir(), "user-data")


if __name__ == "__main__":
    unittest.main()
