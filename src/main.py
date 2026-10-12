from os import path
import os
import sys
import traceback
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import (
    QTranslator,
    QLocale,
)

from resource_path import resource_path
from sc_logging import logger
from http_server import stop_http_server

if "--check-imports" in sys.argv:
    # Importing MainWindow loads the normal camera_thread -> ndi -> cyndilib
    # chain without opening the GUI or touching any camera hardware.
    try:
        from mainwindow import MainWindow  # noqa: F401
        from tesserocr import PyTessBaseAPI

        # Exercise the bundled OCR DLLs and traineddata, not just imports.
        api = PyTessBaseAPI(path=resource_path("tesseract", "tessdata"), lang="eng")
        api.End()
        from dnn_ocr import DnnOcr
        import numpy as np

        # Exercise the ONNX native runtime and packaged model without a camera.
        dnn = DnnOcr(resource_path("models", "en_PP-OCRv4_rec_mobile.onnx"))
        dnn.predict(np.zeros((48, 80, 3), dtype=np.uint8))
        # CI supplies a real scoreboard crop to exercise selective TTA in the
        # frozen executable, including ONNX native libraries and resource paths.
        ocr_check_image = os.environ.get("SCORESIGHT_OCR_CHECK_IMAGE")
        if ocr_check_image:
            import cv2

            crop = cv2.imread(ocr_check_image)
            if crop is None:
                raise RuntimeError("Unable to read the packaged OCR smoke-test input")
            text, _, readings = dnn.recognize(crop)
            if text != "7" or len(readings) != 4:
                raise RuntimeError(f"Packaged TTA check failed: {text!r}, {readings!r}")
    except BaseException:
        import_check_result = traceback.format_exc()
        import_check_exit_code = 1
    else:
        import_check_result = "imports-ok"
        import_check_exit_code = 0

    result_path = os.environ.get("SCORESIGHT_IMPORT_CHECK_RESULT")
    if result_path:
        Path(result_path).write_text(import_check_result, encoding="utf-8")
    else:
        print(import_check_result)
    raise SystemExit(import_check_exit_code)

from mainwindow import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Get system locale
    locale = QLocale.system().name()

    # Load the translation file based on the locale
    translator = QTranslator()
    locale_file = resource_path("translations", f"scoresight_{locale}.qm")
    # check if the file exists
    if not path.exists(locale_file):
        # load the default translation file
        locale_file = resource_path("translations", "scoresight_en_US.qm")
    if translator.load(locale_file):
        app.installTranslator(translator)

    # show the main window
    mainWindow = MainWindow(translator, app)
    mainWindow.show()

    app.exec()
    logger.info("Exiting...")

    stop_http_server()
