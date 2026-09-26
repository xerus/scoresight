# Repository Guidelines

## Project Structure & Module Organization

ScoreSight extracts live scoreboard text using Python, PySide6, OpenCV, and Tesseract. Application code lives in `src/`; `main.py` launches the GUI, while capture, OCR, and output integrations use separate modules. Qt Designer layouts are stored as `src/*.ui`, with generated Python modules named `ui_*.py`.

Regression tests live in `tests/`. Supporting directories include `icons/`, `translations/`, `tesseract/tessdata/` for OCR models, `scripts/` for development utilities, `tools/` for benchmarks, and `docs/` for user guides. Packaging uses `scoresight.spec`, `scoresight.iss`, and `.github/workflows/`.

## Build, Test, and Development Commands

Use Python 3.11 or newer in a virtual environment. Run commands from the repository root:

- `python -m pip install -r requirements.txt -r requirements-dev.txt`: install application dependencies and `prek`. Also install the appropriate Windows or macOS requirements; consult `README.md` for native dependency setup.
- `bash scripts/compile_ui.sh`: regenerate Python UI modules. Windows users can run `./scripts/compile_ui.ps1`.
- `python src/main.py`: launch the application locally.
- `PYTHONPATH=src python -m unittest discover -s tests -v`: run regression tests on Unix-like shells; on Windows, set `PYTHONPATH` to `src` first.
- `prek run --all-files`: run the configured lint and formatting checks used by CI.
- `pyinstaller --clean --noconfirm scoresight.spec`: build on Linux. Append `-- --win` for Windows or `-- --mac_osx` for macOS.

## Coding Style & Naming Conventions

Use four-space indentation, descriptive names, and the surrounding module's conventions. Prefer snake_case for new Python functions and modules, PascalCase for classes, and uppercase constants; preserve existing Qt-style method names. Edit `.ui` sources and regenerate their Python counterparts.

Ruff formatting checks exclude generated `src/ui_*.py` files. The configured Ruff lint hook currently targets only `src/tesseract.py`.

## Testing Guidelines

Use standard-library `unittest`, `test_*.py` filenames, and `test_*` methods. Add focused regression cases for behavior changes, following existing synthetic-image and mocked-dependency examples. No coverage threshold is configured. Manually verify affected GUI or capture behavior when automated tests cannot exercise it.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects, such as “Refresh displayed video resolution.” Keep changes focused. Submit a feature or fix branch through a pull request. Describe the problem, resulting behavior, validation performed, and relevant platform details; link related issues and include screenshots for visible UI changes.
