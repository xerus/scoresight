"""The compiled Czech catalog translates the current interface."""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QTranslator
from PySide6.QtWidgets import QApplication, QMainWindow

from box_settings_ui_handler import BoxSettingsUIHandler
from mainwindow import MainWindow
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
            self.assertEqual(ui.spinBox_hscale.minimum(), 1)
            self.assertEqual(ui.checkBox.text(), "Vynutit formát")
            self.assertEqual(QCoreApplication.translate("MainWindow", "File"), "Soubor")
            handler = BoxSettingsUIHandler(ui)
            self.assertEqual(
                ui.spinBox_hscale.toolTip(),
                "Před OCR upraví šířku znaků. Hodnota 10 zachová původní šířku; nižší hodnoty znaky zúží a vyšší je roztáhnou. V binárním zobrazení se roztažený text ořízne na rámeček.",
            )
            tooltip_host = SimpleNamespace(
                ui=ui,
                tr=lambda source: QCoreApplication.translate("MainWindow", source),
            )
            MainWindow.setGlobalTooltips(tooltip_host)
            self.assertEqual(
                ui.spinBox_leftCrop.toolTip(),
                "Počet pixelů odstraněných z levého okraje obrazu.",
            )
            self.app.removeTranslator(translator)
            handler.setControlTooltips()
            MainWindow.setGlobalTooltips(tooltip_host)
            self.assertEqual(
                ui.spinBox_hscale.toolTip(),
                "Adjust character width before OCR. 10 leaves width unchanged; lower values narrow characters, and higher values spread them apart. Binary View crops widened text to the box.",
            )
            self.assertEqual(
                ui.spinBox_leftCrop.toolTip(),
                "Pixels to remove from the left edge of the image.",
            )
            self.app.installTranslator(translator)
        finally:
            self.app.removeTranslator(translator)


if __name__ == "__main__":
    unittest.main()
