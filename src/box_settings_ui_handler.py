from functools import partial

from defaults import (
    default_info_for_box_name,
    normalize_settings_dict,
    format_prefixes,
)
from storage import TextDetectionTargetMemoryStorage
from ui_mainwindow import Ui_MainWindow
from sc_logging import logger


class BoxSettingsUIHandler:
    def __init__(self, ui: Ui_MainWindow):
        self.ui = ui
        self.setControlTooltips()
        self.boxSettingsUiSetup()
        self.detectionTargetsStorage = TextDetectionTargetMemoryStorage()

    def setControlTooltips(self):
        """Explain OCR controls where users configure each detection box."""
        tooltips = {
            "groupBox_target_settings": (
                "Select a box in the list, then adjust its OCR settings. Start with "
                "the defaults and change one setting at a time while watching the result."
            ),
            "label_13": "Choose whether this field contains digits, time, or general text.",
            "comboBox_fieldType": "Choose whether this field contains digits, time, or general text.",
            "label_2": "A regular expression that accepted OCR text must match. The preset list provides common formats.",
            "lineEdit_format": "A regular expression that accepted OCR text must match. The preset list provides common formats.",
            "comboBox_formatPrefix": "Choose a common value format to fill the Format field, or choose Custom.",
            "checkBox": "Reserved option; currently unavailable.",
            "checkBox_smoothing": "Smooth readings over time to reduce flicker. This can make updates slower to reflect.",
            "checkBox_ordinalIndicator": "Append an ordinal suffix to numeric values, for example 1st or 2nd.",
            "checkBox_skip_empty": "Do not send or display a new value when OCR reads nothing.",
            "checkBox_skip_similar_image": "Skip OCR when this box looks almost unchanged from its previous image. Useful for reducing repeated work on static displays.",
            "checkBox_autocrop": "Trim blank margins around the contents before OCR. Turn off if edge pixels or punctuation are being cut off.",
            "checkBox_removeLeadingZeros": "Remove zeros at the start of numeric readings; for example, 007 becomes 7.",
            "checkBox_rescalePatch": "Resize only this box to 35 pixels high before Tesseract reads it. This reduces OCR input size and can help small text; dense details may be lost.",
            "checkBox_normWHRatio": "Resize the patch toward a 1:2 width-to-height ratio before OCR. Use only when the characters are unusually wide or narrow.",
            "checkBox_invertPatch": "Invert light and dark pixels in this box. Use when the current foreground/background polarity gives poor OCR.",
            "checkBox_dotDetector": "Count bright blobs or dots instead of recognizing characters. Intended for displays made of separate indicator dots.",
            "checkBox_templatefield": "Make this a derived field whose value is assembled from other fields using a template.",
            "lineEdit_templatefield": "Enter a template using other field values, for example {{Home Score}}.",
            "checkBox_compositeBox": "Read separate character sub-boxes and combine their results. Set up the sub-boxes for this field first.",
            "label_binarizationMethod": "Choose how the image is turned into black and white before OCR.",
            "comboBox_binarizationMethod": (
                "Global uses one Otsu threshold computed from the selected boxes; "
                "Local computes a separate Otsu threshold for this box; No Binarization "
                "keeps grayscale; Adaptive adjusts the threshold within this box; "
                "OpenCV LED Mask selects bright red/yellow LEDs by color and removes "
                "most of the surrounding scoreboard. It works best with warm-colored "
                "LEDs; other display colors may be removed."
            ),
            "label_4": (
                "Remove small isolated blobs before OCR. 0 disables cleanup; 100 "
                "removes components smaller than about 5% of this box's area. Increase "
                "gradually: high values can erase dots or parts of digits."
            ),
            "spinBox_cleanup": (
                "Remove small isolated blobs before OCR. 0 disables cleanup; 100 "
                "removes components smaller than about 5% of this box's area. Increase "
                "gradually: high values can erase dots or parts of digits."
            ),
            "label_9": "Expand white strokes with a 3x3 kernel. Helps broken segments, but too much joins nearby LEDs; use 0 to disable.",
            "spinBox_dilate": "Expand white strokes with a 3x3 kernel. Helps broken segments, but too much joins nearby LEDs; use 0 to disable.",
            "label_15": "Adjust character height. 10 keeps the original height; lower values make characters shorter.",
            "spinBox_vscale": "Adjust character height. 10 keeps the original height; lower values make characters shorter.",
            "label_14": "Shear the image sideways to compensate for slanted characters. 0 leaves it unchanged.",
            "spinBox_skew": "Shear the image sideways to compensate for slanted characters. 0 leaves it unchanged.",
            "label_3": "Reject OCR readings below this confidence percentage. Raise it to filter uncertain readings; lower it if valid readings are rejected.",
            "spinBox_conf_thresh": "Reject OCR readings below this confidence percentage. Raise it to filter uncertain readings; lower it if valid readings are rejected.",
        }
        for object_name, tooltip in tooltips.items():
            widget = getattr(self.ui, object_name, None)
            if widget is not None:
                widget.setToolTip(tooltip)

    def editSettings(self, settingsMutatorCallback):
        # update the selected item's settings in the detectionTargetsStorage
        item = self.ui.tableWidget_boxes.currentItem()
        if item is None:
            logger.info("no item selected")
            return
        item_name = item.text()
        item_obj = self.detectionTargetsStorage.find_item_by_name(item_name)
        if item_obj is None:
            logger.info("item not found: %s", item_name)
            return
        item_obj = settingsMutatorCallback(item_obj)
        self.detectionTargetsStorage.edit_item(item_name, item_obj)

    def restoreDefaults(self):
        # restore the default settings for the selected item
        def restoreDefaultsSettings(item_obj):
            info = default_info_for_box_name(item_obj.name)
            item_obj.settings = normalize_settings_dict({}, info)
            return item_obj

        self.editSettings(restoreDefaultsSettings)
        self.populateSettings(self.ui.tableWidget_boxes.currentItem().text())

    def confThreshChanged(self):
        self.genericSettingsChanged(
            "conf_thresh", float(self.ui.spinBox_conf_thresh.value()) / 100.0
        )

    def cleanupThreshChanged(self):
        self.genericSettingsChanged(
            "cleanup_thresh", float(self.ui.spinBox_cleanup.value()) / 100.0
        )

    def formatPrefixChanged(self, index):
        if index == 12:
            return  # do nothing if "Select Preset" is selected
        # based on the selected index, set the format prefix
        # change lineEdit_format to the selected format prefix
        self.ui.lineEdit_format.setText(format_prefixes[index])

    def genericSettingsChanged(self, settingName, value):
        def editGenericSettings(item_obj):
            item_obj.settings[settingName] = value
            return item_obj

        self.editSettings(editGenericSettings)

    def boxSettingsUiSetup(self):
        self.ui.pushButton_restoreDefaults.clicked.connect(self.restoreDefaults)
        self.ui.checkBox_smoothing.toggled.connect(
            partial(self.genericSettingsChanged, "smoothing")
        )
        self.ui.checkBox_skip_empty.toggled.connect(
            partial(self.genericSettingsChanged, "skip_empty")
        )
        self.ui.spinBox_conf_thresh.valueChanged.connect(self.confThreshChanged)
        self.ui.lineEdit_format.textChanged.connect(
            partial(self.genericSettingsChanged, "format_regex")
        )
        self.ui.comboBox_fieldType.currentIndexChanged.connect(
            partial(self.genericSettingsChanged, "type")
        )
        self.ui.checkBox_skip_similar_image.toggled.connect(
            partial(self.genericSettingsChanged, "skip_similar_image")
        )
        self.ui.checkBox_autocrop.toggled.connect(
            partial(self.genericSettingsChanged, "autocrop")
        )
        self.ui.spinBox_cleanup.valueChanged.connect(self.cleanupThreshChanged)
        self.ui.spinBox_dilate.valueChanged.connect(
            partial(self.genericSettingsChanged, "dilate")
        )
        self.ui.spinBox_skew.valueChanged.connect(
            partial(self.genericSettingsChanged, "skew")
        )
        self.ui.spinBox_vscale.valueChanged.connect(
            partial(self.genericSettingsChanged, "vscale")
        )
        self.ui.checkBox_removeLeadingZeros.toggled.connect(
            partial(self.genericSettingsChanged, "remove_leading_zeros")
        )
        self.ui.checkBox_rescalePatch.toggled.connect(
            partial(self.genericSettingsChanged, "rescale_patch")
        )
        self.ui.checkBox_normWHRatio.toggled.connect(
            partial(self.genericSettingsChanged, "normalize_wh_ratio")
        )
        self.ui.checkBox_invertPatch.toggled.connect(
            partial(self.genericSettingsChanged, "invert_patch")
        )
        self.ui.checkBox_dotDetector.toggled.connect(
            partial(self.genericSettingsChanged, "dot_detector")
        )
        self.ui.checkBox_ordinalIndicator.toggled.connect(
            partial(self.genericSettingsChanged, "ordinal_indicator")
        )
        self.ui.comboBox_binarizationMethod.currentIndexChanged.connect(
            partial(self.genericSettingsChanged, "binarization_method")
        )
        self.ui.lineEdit_templatefield.textChanged.connect(
            partial(self.genericSettingsChanged, "templatefield_text")
        )
        self.ui.checkBox_compositeBox.toggled.connect(
            partial(self.genericSettingsChanged, "composite_box")
        )
        self.ui.comboBox_formatPrefix.currentIndexChanged.connect(
            self.formatPrefixChanged
        )

    def populateSettings(self, name):
        self.ui.lineEdit_format.blockSignals(True)
        self.ui.comboBox_fieldType.blockSignals(True)
        self.ui.checkBox_smoothing.blockSignals(True)
        self.ui.checkBox_skip_empty.blockSignals(True)
        self.ui.spinBox_conf_thresh.blockSignals(True)
        self.ui.checkBox_autocrop.blockSignals(True)
        self.ui.checkBox_skip_similar_image.blockSignals(True)
        self.ui.spinBox_cleanup.blockSignals(True)
        self.ui.spinBox_dilate.blockSignals(True)
        self.ui.spinBox_skew.blockSignals(True)
        self.ui.spinBox_vscale.blockSignals(True)
        self.ui.checkBox_removeLeadingZeros.blockSignals(True)
        self.ui.checkBox_rescalePatch.blockSignals(True)
        self.ui.checkBox_normWHRatio.blockSignals(True)
        self.ui.checkBox_invertPatch.blockSignals(True)
        self.ui.checkBox_ordinalIndicator.blockSignals(True)
        self.ui.comboBox_binarizationMethod.blockSignals(True)
        self.ui.comboBox_formatPrefix.blockSignals(True)
        self.ui.checkBox_templatefield.blockSignals(True)
        self.ui.lineEdit_templatefield.blockSignals(True)
        self.ui.checkBox_compositeBox.blockSignals(True)

        # populate the settings from the detectionTargetsStorage
        item_obj = self.detectionTargetsStorage.find_item_by_name(name)
        if item_obj is None:
            self.ui.lineEdit_format.setText("")
            self.ui.comboBox_fieldType.setCurrentIndex(0)
            self.ui.checkBox_smoothing.setChecked(True)
            self.ui.checkBox_skip_empty.setChecked(True)
            self.ui.spinBox_conf_thresh.setValue(50)
            self.ui.checkBox_autocrop.setChecked(False)
            self.ui.checkBox_skip_similar_image.setChecked(False)
            self.ui.spinBox_cleanup.setValue(0)
            self.ui.spinBox_dilate.setValue(1)
            self.ui.spinBox_skew.setValue(0)
            self.ui.spinBox_vscale.setValue(10)
            self.ui.label_selectedInfo.setText("")
            self.ui.checkBox_removeLeadingZeros.setChecked(False)
            self.ui.checkBox_rescalePatch.setChecked(False)
            self.ui.checkBox_normWHRatio.setChecked(False)
            self.ui.checkBox_invertPatch.setChecked(False)
            self.ui.checkBox_ordinalIndicator.setChecked(False)
            self.ui.comboBox_binarizationMethod.setCurrentIndex(0)
            self.ui.checkBox_templatefield.setChecked(False)
            self.ui.lineEdit_templatefield.setText("")
            self.ui.checkBox_compositeBox.setChecked(False)
        else:
            item_obj.settings = normalize_settings_dict(
                item_obj.settings, default_info_for_box_name(item_obj.name)
            )
            self.ui.label_selectedInfo.setText(f"{item_obj.name}")
            self.ui.lineEdit_format.setText(item_obj.settings["format_regex"])
            self.ui.comboBox_fieldType.setCurrentIndex(item_obj.settings["type"])
            self.ui.checkBox_smoothing.setChecked(item_obj.settings["smoothing"])
            self.ui.checkBox_skip_empty.setChecked(item_obj.settings["skip_empty"])
            self.ui.spinBox_conf_thresh.setValue(
                int(item_obj.settings["conf_thresh"] * 100)
            )
            self.ui.checkBox_autocrop.setChecked(item_obj.settings["autocrop"])
            self.ui.checkBox_skip_similar_image.setChecked(
                item_obj.settings["skip_similar_image"]
            )
            self.ui.spinBox_cleanup.setValue(
                int(item_obj.settings["cleanup_thresh"] * 100)
            )
            self.ui.spinBox_dilate.setValue(item_obj.settings["dilate"])
            self.ui.spinBox_skew.setValue(item_obj.settings["skew"])
            self.ui.spinBox_vscale.setValue(item_obj.settings["vscale"])
            self.ui.checkBox_removeLeadingZeros.setChecked(
                item_obj.settings["remove_leading_zeros"]
            )
            self.ui.checkBox_rescalePatch.setChecked(item_obj.settings["rescale_patch"])
            self.ui.checkBox_normWHRatio.setChecked(
                item_obj.settings["normalize_wh_ratio"]
            )
            self.ui.checkBox_invertPatch.setChecked(item_obj.settings["invert_patch"])
            self.ui.checkBox_dotDetector.setChecked(item_obj.settings["dot_detector"])
            self.ui.checkBox_ordinalIndicator.setChecked(
                item_obj.settings["ordinal_indicator"]
            )
            self.ui.comboBox_binarizationMethod.setCurrentIndex(
                item_obj.settings["binarization_method"]
            )
            self.ui.checkBox_templatefield.setChecked(
                item_obj.settings["templatefield"]
            )
            self.ui.lineEdit_templatefield.setText(
                item_obj.settings["templatefield_text"]
            )
            self.ui.checkBox_compositeBox.setChecked(item_obj.settings["composite_box"])

        self.ui.comboBox_formatPrefix.setCurrentIndex(12)

        self.ui.lineEdit_format.blockSignals(False)
        self.ui.comboBox_fieldType.blockSignals(False)
        self.ui.checkBox_smoothing.blockSignals(False)
        self.ui.checkBox_skip_empty.blockSignals(False)
        self.ui.spinBox_conf_thresh.blockSignals(False)
        self.ui.checkBox_autocrop.blockSignals(False)
        self.ui.checkBox_skip_similar_image.blockSignals(False)
        self.ui.spinBox_cleanup.blockSignals(False)
        self.ui.spinBox_dilate.blockSignals(False)
        self.ui.spinBox_skew.blockSignals(False)
        self.ui.spinBox_vscale.blockSignals(False)
        self.ui.checkBox_removeLeadingZeros.blockSignals(False)
        self.ui.checkBox_rescalePatch.blockSignals(False)
        self.ui.checkBox_normWHRatio.blockSignals(False)
        self.ui.checkBox_invertPatch.blockSignals(False)
        self.ui.checkBox_ordinalIndicator.blockSignals(False)
        self.ui.comboBox_binarizationMethod.blockSignals(False)
        self.ui.comboBox_formatPrefix.blockSignals(False)
        self.ui.checkBox_templatefield.blockSignals(False)
        self.ui.lineEdit_templatefield.blockSignals(False)
        self.ui.checkBox_compositeBox.blockSignals(False)
