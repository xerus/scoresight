"""The compiled Czech catalog translates the current interface."""

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QTranslator
from PySide6.QtWidgets import QApplication, QMainWindow

from ui_mainwindow import Ui_MainWindow


class CzechTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_compiled_catalog_translates_preview_and_menus(self):
        catalog = (
            Path(__file__).resolve().parents[1] / "translations" / "scoresight_cs_CZ.qm"
        )
        translator = QTranslator()
        self.assertTrue(translator.load(str(catalog)))
        self.app.installTranslator(translator)
        try:
            window = QMainWindow()
            ui = Ui_MainWindow()
            ui.setupUi(window)
            self.assertEqual(
                ui.comboBox_boxDisplayStyle.itemText(4), "Výsledky (bez názvů)"
            )
            self.assertEqual(ui.label_hscale.text(), "Vodor. měřítko")
            self.assertEqual(ui.spinBox_hscale.value(), 10)
            self.assertEqual(ui.checkBox.text(), "Vynutit formát")
            self.assertEqual(QCoreApplication.translate("MainWindow", "File"), "Soubor")
        finally:
            self.app.removeTranslator(translator)


if __name__ == "__main__":
    unittest.main()
