from defaults import FieldType
from os import path
from PIL import Image
from PySide6.QtCore import QRectF
from tesserocr import PyTessBaseAPI, RIL, iterate_level
from threading import Lock
import cv2
import numpy as np
import re

from seven_segment import read_score, threshold_led_colors
from resource_path import resource_path
from storage import fetch_data
from text_detection_target import (
    TextDetectionResult,
    TextDetectionTarget,
    TextDetectionTargetWithResult,
)
from sc_logging import logger


def autocrop(image_in):
    image = image_in.copy()
    # check if this image is black-on-white or white-on-black by looking at the first few pixels
    if np.sum(image[0:5, 0:5]) > 0:
        # black-on-white
        # invert the image
        image = 255 - image

    # find the first row that has a pixel
    first_row = 0
    for row in range(image.shape[0]):
        if np.sum(image[row, :]) > 0:
            first_row = row
            break
    # find the last row that has a pixel
    last_row = image.shape[0] - 1
    for row in range(image.shape[0] - 1, -1, -1):
        if np.sum(image[row, :]) > 0:
            last_row = row
            break
    # find the first column that has a pixel
    first_col = 0
    for col in range(image.shape[1]):
        if np.sum(image[:, col]) > 0:
            first_col = col
            break
    # find the last column that has a pixel
    last_col = image.shape[1] - 1
    for col in range(image.shape[1] - 1, -1, -1):
        if np.sum(image[:, col]) > 0:
            last_col = col
            break
    # leave a 10 pixel border on each side
    first_row = max(0, first_row - 10)
    last_row = min(image.shape[0] - 1, last_row + 10)
    first_col = max(0, first_col - 10)
    last_col = min(image.shape[1] - 1, last_col + 10)
    return image_in[first_row:last_row, first_col:last_col], (
        first_row,
        last_row,
        first_col,
        last_col,
    )


def add_ordinal_indicator(text):
    if text == "":
        return ""
    if text.endswith("1") and text != "11":
        return text + "st"
    elif text.endswith("2") and text != "12":
        return text + "nd"
    elif text.endswith("3") and text != "13":
        return text + "rd"
    else:
        return text + "th"


def is_valid_regex(pattern):
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


class TextDetector:
    # model name enum: daktronics=0, scoreboard_general=1
    class OcrModelIndex:
        DAKTRONICS = 0
        SCOREBOARD_GENERAL = 1
        GENERAL_ENGLISH = 2
        SCOREBOARD_GENERAL_LARGE = 3
        # Index 4 remains the external Tesseract model file picker.
        OPENCV_SEVEN_SEGMENT = 5

    class BinarizationMethod:
        GLOBAL = 0
        NO_BINARIZATION = 1
        LOCAL = 2
        ADAPTIVE = 3
        OPENCV_LED_MASK = 4

    def __init__(self):
        self.api_lock = Lock()
        self.api = None
        self.ocr_model_index = None
        self.setOcrModel(fetch_data("scoresight.json", "ocr_model", 1))
        if (
            self.api is None
            and self.ocr_model_index != self.OcrModelIndex.OPENCV_SEVEN_SEGMENT
        ):
            self.setOcrModel(self.OcrModelIndex.SCOREBOARD_GENERAL)

    def setOcrModel(self, ocrModelIndex: OcrModelIndex | int | str | None = None):
        if ocrModelIndex == self.OcrModelIndex.OPENCV_SEVEN_SEGMENT:
            with self.api_lock:
                if self.api is not None:
                    self.api.End()
                    self.api = None
                self.ocr_model_index = ocrModelIndex
            return
        ocr_model = None
        model_folder = resource_path("tesseract", "tessdata")
        if ocrModelIndex == TextDetector.OcrModelIndex.DAKTRONICS:
            ocr_model = "daktronics"
        elif ocrModelIndex == TextDetector.OcrModelIndex.SCOREBOARD_GENERAL:
            ocr_model = "scoreboard_general"
        elif ocrModelIndex == TextDetector.OcrModelIndex.GENERAL_ENGLISH:
            ocr_model = "eng"
        elif ocrModelIndex == TextDetector.OcrModelIndex.SCOREBOARD_GENERAL_LARGE:
            ocr_model = "scoreboard_general_large"
        elif isinstance(ocrModelIndex, str):
            # check the model file exists at the path
            if path.exists(ocrModelIndex):
                # Take the folder as the tessdata folder
                model_folder = path.dirname(ocrModelIndex)
                # Take the model name without extension as the "language"
                ocr_model = path.basename(ocrModelIndex)
                ocr_model = path.splitext(ocr_model)[0]

        if ocr_model is None:
            return

        logger.info(f"Setting OCR model to {ocr_model} from {model_folder}")

        with self.api_lock:
            if self.api is not None:
                self.api.End()
                self.api = None
            self.api = PyTessBaseAPI(
                path=model_folder,
                lang=ocr_model,
            )
            self.ocr_model_index = ocrModelIndex
            # single word PSM
            self.api.SetPageSegMode(8)
            self.api.SetVariable("load_system_dawg", "F")
            self.api.SetVariable("load_freq_dawg", "F")

    def detect_text(self, image):
        if image is None:
            return ""
        if not isinstance(image, np.ndarray):
            return ""
        # check the image has rows and columns
        if len(image.shape) < 2 or image.shape[0] < 1 or image.shape[1] < 1:
            return ""
        if self.ocr_model_index == self.OcrModelIndex.OPENCV_SEVEN_SEGMENT:
            return read_score(image)[0] or ""
        pilimage = Image.fromarray(image)
        text = ""
        with self.api_lock:
            self.api.SetImage(pilimage)
            text = self.api.GetUTF8Text()
        return text.strip()

    def _detect_seven_segment(self, color, binary, rects):
        """Use the transformed BGR frame; preserve blank and rejected states."""
        results = []
        states = TextDetectionTargetWithResult.ResultState
        valid_frame = (
            isinstance(color, np.ndarray) and color.ndim == 3 and color.shape[2] == 3
        )
        for rect in rects:
            result = TextDetectionResult("", states.FailedFilter, None)
            results.append(result)
            if rect is None or not valid_frame:
                continue
            settings = rect.settings or {}
            # This calibrated score reader does not parse clocks or general text.
            if settings.get("type", FieldType.NUMBER) != FieldType.NUMBER:
                continue
            x, y = int(rect.x()), int(rect.y())
            right, bottom = int(rect.x() + rect.width()), int(rect.y() + rect.height())
            if (
                x < 0
                or y < 0
                or right > color.shape[1]
                or bottom > color.shape[0]
                or right <= x
                or bottom <= y
            ):
                continue
            text, mask, details = read_score(color[y:bottom, x:right])
            result.extra = {"segment_slots": details, "engine": "opencv_seven_segment"}
            if (
                mask is not None
                and isinstance(binary, np.ndarray)
                and binary.shape == color.shape[:2]
            ):
                binary[y:bottom, x:right] = mask
            # Do not feed geometry readings into per-character Tesseract smoothing.
            rect.ocrResultPerCharacterSmoother.clear()
            if text is None:
                continue
            result.text = text
            if text == "":
                result.state = states.Empty
                continue
            if settings.get("remove_leading_zeros"):
                result.text = text.lstrip("0") or "0"
            pattern = settings.get("format_regex")
            if pattern and (
                not is_valid_regex(pattern) or not re.fullmatch(pattern, result.text)
            ):
                continue
            if settings.get("ordinal_indicator"):
                result.text = add_ordinal_indicator(result.text)
            result.state = states.Success
        return results

    def detect_multi_text(
        self, binary, gray, rects: list[TextDetectionTarget], color=None
    ) -> list[TextDetectionResult]:
        if self.ocr_model_index == self.OcrModelIndex.OPENCV_SEVEN_SEGMENT:
            return self._detect_seven_segment(color, binary, rects)
        color_valid = (
            isinstance(color, np.ndarray)
            and color.ndim == 3
            and color.shape[2] == 3
            and color.shape[0] > 0
            and color.shape[1] > 0
        )
        gray_valid = (
            isinstance(gray, np.ndarray)
            and gray.ndim == 2
            and gray.shape[0] > 0
            and gray.shape[1] > 0
        )
        binary_valid = (
            isinstance(binary, np.ndarray)
            and binary.ndim == 2
            and binary.shape[0] > 0
            and binary.shape[1] > 0
        )
        if not (color_valid or gray_valid or binary_valid):
            return []

        frame_height, frame_width = (
            color.shape[:2]
            if color_valid
            else gray.shape[:2]
            if gray_valid
            else binary.shape[:2]
        )

        def gray_roi_for(rect):
            x = max(0, int(rect.x()))
            y = max(0, int(rect.y()))
            right = min(frame_width, int(rect.x() + rect.width()))
            bottom = min(frame_height, int(rect.y() + rect.height()))
            if right <= x or bottom <= y:
                return None
            if color_valid:
                return cv2.cvtColor(color[y:bottom, x:right], cv2.COLOR_BGR2GRAY)
            if gray_valid:
                return gray[y:bottom, x:right]
            return binary[y:bottom, x:right]

        roi_gray = {}
        global_rois = []
        for index, rect in enumerate(rects):
            if (
                rect is None
                or rect.x() < 0
                or rect.y() < 0
                or rect.width() < 1
                or rect.height() < 1
            ):
                continue
            patch = gray_roi_for(rect)
            if patch is None or patch.size == 0:
                continue
            roi_gray[index] = patch
            settings = rect.settings or {}
            method = settings.get("binarization_method", self.BinarizationMethod.GLOBAL)
            if method == self.BinarizationMethod.GLOBAL:
                global_rois.append(patch.reshape(-1))

        # Preserve one shared global threshold while computing it only from
        # selected OCR regions. Local mode computes an independent threshold
        # for each selected region.
        global_threshold = 127
        if global_rois:
            pixels = np.concatenate(global_rois).reshape(-1, 1)
            global_threshold = cv2.threshold(
                pixels, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )[0]

        texts = []
        for rect_index, rect in enumerate(rects):
            effectiveRect = None
            scale_x = 1.0
            scale_y = 1.0

            if (
                rect is None
                or rect.x() < 0
                or rect.y() < 0
                or rect.width() < 1
                or rect.height() < 1
            ):
                texts.append(
                    TextDetectionResult(
                        "", TextDetectionTargetWithResult.ResultState.Empty, None
                    )
                )
                continue

            if rect.x() >= frame_width:
                # move the rect inside the image
                rect.setX(frame_width - rect.width())
            if rect.y() >= frame_height:
                # move the rect inside the image
                rect.setY(frame_height - rect.height())
            if rect.x() + rect.width() > frame_width:
                rect.setWidth(frame_width - rect.x())
            if rect.y() + rect.height() > frame_height:
                rect.setHeight(frame_height - rect.y())

            x, y = int(rect.x()), int(rect.y())
            right = int(rect.x() + rect.width())
            bottom = int(rect.y() + rect.height())
            graycrop = roi_gray.get(rect_index)
            if graycrop is None or graycrop.shape != (bottom - y, right - x):
                graycrop = gray_roi_for(rect)
            if graycrop is None or graycrop.size == 0:
                texts.append(
                    TextDetectionResult(
                        "", TextDetectionTargetWithResult.ResultState.Empty, None
                    )
                )
                continue

            settings = rect.settings or {}
            binarization_method = settings.get(
                "binarization_method", self.BinarizationMethod.GLOBAL
            )
            if binarization_method == self.BinarizationMethod.NO_BINARIZATION:
                imagecrop = graycrop.copy()
            elif binarization_method == self.BinarizationMethod.LOCAL:
                # Otsu's threshold is calculated independently for this box.
                _, imagecrop = cv2.threshold(
                    graycrop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                )
            elif binarization_method == self.BinarizationMethod.ADAPTIVE:
                # Keep the neighborhood odd and no larger than the crop.
                max_block = min(graycrop.shape)
                if max_block % 2 == 0:
                    max_block -= 1
                if max_block < 3:
                    _, imagecrop = cv2.threshold(
                        graycrop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                    )
                else:
                    block_size = max(3, min(int(graycrop.size * 0.01) | 1, max_block))
                    imagecrop = cv2.adaptiveThreshold(
                        graycrop,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        block_size,
                        2,
                    )
            elif binarization_method == self.BinarizationMethod.OPENCV_LED_MASK:
                # Apply the calibrated OpenCV HSV mask to this OCR region only.
                # If only grayscale input is available, keep OCR functional by
                # falling back to a local Otsu threshold.
                if color_valid:
                    imagecrop = threshold_led_colors(color[y:bottom, x:right])
                else:
                    _, imagecrop = cv2.threshold(
                        graycrop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                    )
            else:
                # Global mode shares an Otsu threshold across selected boxes;
                # no pixels outside OCR targets contribute to the threshold.
                _, imagecrop = cv2.threshold(
                    graycrop,
                    global_threshold,
                    255,
                    cv2.THRESH_BINARY,
                )

            # Binary View and training export need a full-size image; normal
            # OCR operation leaves it unset and only processes the selected ROI.
            if isinstance(binary, np.ndarray):
                binary[y:bottom, x:right] = imagecrop

            if (
                rect.settings is not None
                and "cleanup_thresh" in rect.settings
                and rect.settings["cleanup_thresh"] > 0
            ):
                # cleanup image from small components: find contours and remove small ones
                contours, _ = cv2.findContours(
                    imagecrop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                # cleanup_thresh is [0, 1.0], convert to [0, 0.05]
                cleanup_thresh = rect.settings["cleanup_thresh"] * 0.05
                img_area_thresh = (
                    imagecrop.shape[0] * imagecrop.shape[1] * cleanup_thresh
                )
                for contour in contours:
                    if cv2.contourArea(contour) < img_area_thresh:
                        cv2.drawContours(imagecrop, [contour], 0, 0, -1)

            if (
                rect.settings is not None
                and "vscale" in rect.settings
                and rect.settings["vscale"] != 10
            ):
                # vertical scale the image
                # the vscale input is in the range [1, 10] where 10 is the default (1:1)
                # scale the image in the y direction about the center
                rows, cols = imagecrop.shape
                # calculate the target height
                target_height = int(rows * (rect.settings["vscale"] / 10.0))
                scaled = cv2.resize(
                    imagecrop, (cols, target_height), 0, 0, cv2.INTER_AREA
                )
                # add padding to the top and bottom
                pad_top = (rows - target_height) // 2
                pad_bottom = rows - target_height - pad_top
                scaled = cv2.copyMakeBorder(
                    scaled, pad_top, pad_bottom, 0, 0, cv2.BORDER_REPLICATE
                )
                # make sure the image is the same size as the original
                scaled = scaled[:rows, :]
                # copy back into the optional binary display
                if isinstance(binary, np.ndarray):
                    binary[y:bottom, x:right] = scaled
                imagecrop = scaled

            if (
                rect.settings is not None
                and "skew" in rect.settings
                and rect.settings["skew"] != 0
            ):
                # skew the image in the x direction about the center
                rows, cols = imagecrop.shape
                # identity 2x2 matrix
                M = np.float32([[1, 0, 0], [0, 1, 0]])
                # add skew factor to matrix
                M[0, 1] = rect.settings["skew"] / 40.0
                try:
                    skewed = cv2.warpAffine(imagecrop, M, (cols, rows))
                    if isinstance(binary, np.ndarray):
                        binary[y:bottom, x:right] = skewed
                    imagecrop = skewed
                except Exception:
                    pass

            if (
                rect.settings is not None
                and "dilate" in rect.settings
                and rect.settings["dilate"] > 0
                and imagecrop.shape[0] > 0
                and imagecrop.shape[1] > 0
            ):
                # dilate the image
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(
                    imagecrop.copy(),
                    kernel,
                    iterations=int(rect.settings["dilate"]),
                )
                # copy back into image crop
                if isinstance(binary, np.ndarray):
                    binary[y:bottom, x:right] = dilated

            if (
                rect.settings is not None
                and "invert_patch" in rect.settings
                and rect.settings["invert_patch"]
            ):
                # invert the image
                imagecrop = 255 - imagecrop

            if (
                rect.settings is not None
                and "skip_similar_image" in rect.settings
                and rect.settings["skip_similar_image"]
            ):
                # compare the image with the last image
                if (
                    rect.last_image is not None
                    and rect.last_image.shape == imagecrop.shape
                ):
                    # check if the difference is less than 5%
                    diff = cv2.absdiff(rect.last_image, imagecrop)
                    diff = diff.astype(np.float32)
                    diff = diff / 255.0
                    diff = diff.sum() / (imagecrop.shape[0] * imagecrop.shape[1])
                    if diff < 0.05:
                        # skip this image
                        texts.append(
                            TextDetectionResult(
                                "SIM",
                                TextDetectionTargetWithResult.ResultState.FailedFilter,
                                effectiveRect,
                            )
                        )
                        continue
                rect.last_image = imagecrop.copy()

            if (
                rect.settings is not None
                and "autocrop" in rect.settings
                and rect.settings["autocrop"]
            ):
                # auto crop the binary image around the text
                imagecrop, (first_row, last_row, first_col, last_col) = autocrop(
                    imagecrop
                )
                effectiveRect = QRectF(
                    first_col,
                    first_row,
                    last_col - first_col,
                    last_row - first_row,
                )

            # check if image is size 0
            if imagecrop.shape[0] == 0 or imagecrop.shape[1] == 0:
                texts.append(
                    TextDetectionResult(
                        "",
                        TextDetectionTargetWithResult.ResultState.Empty,
                        effectiveRect,
                    )
                )
                continue

            if (
                rect.settings is not None
                and "rescale_patch" in rect.settings
                and rect.settings["rescale_patch"]
            ):
                # rescale the image to 35 pixels height
                scale_x = 35 / imagecrop.shape[0]
                scale_y = scale_x

            if (
                rect.settings is not None
                and "normalize_wh_ratio" in rect.settings
                and rect.settings["normalize_wh_ratio"]
                and "median_wh_ratio" in rect.settings
                and rect.settings["median_wh_ratio"] > 0
            ):
                # rescale the image in x or in y such that the width-to-height ratio is 0.5
                scale_x *= 0.5 / rect.settings["median_wh_ratio"]

            # Widen close characters in the OCR patch without moving the target box.
            if rect.settings is not None:
                scale_x *= rect.settings.get("hscale", 10) / 10.0

            if scale_x != 1.0 or scale_y != 1.0:
                imagecrop = cv2.resize(
                    imagecrop,
                    None,
                    fx=scale_x,
                    fy=scale_y,
                    interpolation=cv2.INTER_AREA,
                )

            # if dot detector count the blobs in the patch
            if (
                rect.settings is not None
                and "dot_detector" in rect.settings
                and rect.settings["dot_detector"]
            ):
                # find the contours
                contours, _ = cv2.findContours(
                    imagecrop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                # count the number of contours
                count = 0
                for contour in contours:
                    if cv2.contourArea(contour) > 5:
                        count += 1

                texts.append(
                    TextDetectionResult(
                        str(count),
                        TextDetectionTargetWithResult.ResultState.Success,
                        effectiveRect,
                    )
                )
                continue

            try:
                pilimage = Image.fromarray(imagecrop)
                with self.api_lock:
                    self.api.SetImage(pilimage)
            except Exception:
                texts.append(
                    TextDetectionResult(
                        "", TextDetectionTargetWithResult.ResultState.Empty, None
                    )
                )
                continue

            if rect.settings["type"] == FieldType.NUMBER:
                with self.api_lock:
                    self.api.SetVariable("tessedit_char_whitelist", "0123456789")
            elif rect.settings["type"] == FieldType.TIME:
                with self.api_lock:
                    self.api.SetVariable("tessedit_char_whitelist", "0123456789:.")
            else:  # general
                with self.api_lock:
                    self.api.SetVariable(
                        "tessedit_char_whitelist",
                        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;:!?-_()[]{}<>@#$%^&*+=|\\~`\"'",
                    )

            text = ""
            extras = {}
            with self.api_lock:
                text = self.api.GetUTF8Text().strip()
                if text != "":
                    # get the per-character boxes using an iterator with RIL_SYMBOL level
                    it = self.api.GetIterator()
                    extras["boxes"] = []
                    wh_ratios = []
                    for w in iterate_level(it, RIL.SYMBOL):
                        char = w.GetUTF8Text(RIL.SYMBOL)
                        box_tuple = w.BoundingBox(RIL.SYMBOL)
                        if (
                            box_tuple is None
                            or char is None
                            or char == ""
                            or len(box_tuple) != 4
                        ):
                            continue
                        box = {
                            "x": box_tuple[0],
                            "y": box_tuple[1],
                            "w": box_tuple[2] - box_tuple[0],
                            "h": box_tuple[3] - box_tuple[1],
                        }
                        # box is a dict with x, y, w and h
                        if scale_x != 1.0 or scale_y != 1.0:
                            box["x"] = int(box["x"] / scale_x)
                            box["y"] = int(box["y"] / scale_y)
                            box["w"] = int(box["w"] / scale_x)
                            box["h"] = int(box["h"] / scale_y)
                        if effectiveRect is not None:
                            box["x"] = int(box["x"] + effectiveRect.x())
                            box["y"] = int(box["y"] + effectiveRect.y())
                        extras["boxes"].append(box)
                        # if char is a "wide character" (like 0,2,3,4,5,6,7,8,9), add the width-to-height ratio
                        if char in "023456789" and box["h"] > 0:
                            wh_ratios.append(box["w"] / box["h"])
                    if (
                        "normalize_wh_ratio" in rect.settings
                        and rect.settings["normalize_wh_ratio"]
                        and "median_wh_ratio" not in rect.settings
                        and len(wh_ratios) > 0
                    ):
                        rect.settings["median_wh_ratio"] = np.median(wh_ratios)

            textstate = TextDetectionTargetWithResult.ResultState.Success
            if rect.settings is not None:
                if "format_regex" in rect.settings:
                    # validate the regex format is valid
                    if is_valid_regex(rect.settings["format_regex"]):
                        # check the text matches the regex fully
                        if not re.fullmatch(rect.settings["format_regex"], text):
                            textstate = (
                                TextDetectionTargetWithResult.ResultState.FailedFilter
                            )
                if "conf_thresh" in rect.settings:
                    with self.api_lock:
                        mean_conf = self.api.MeanTextConf() / 100.0
                    if mean_conf < rect.settings["conf_thresh"]:
                        textstate = (
                            TextDetectionTargetWithResult.ResultState.FailedFilter
                        )
                if "smoothing" in rect.settings:
                    if rect.settings["smoothing"]:
                        smoother = rect.ocrResultPerCharacterSmoother
                        if (
                            textstate
                            == TextDetectionTargetWithResult.ResultState.Success
                            and text
                        ):
                            # Low-confidence and format-rejected frames must not
                            # contaminate the history used to stabilize valid OCR.
                            text = smoother.get_smoothed_result(text) or ""
                            if (
                                "format_regex" in rect.settings
                                and is_valid_regex(rect.settings["format_regex"])
                                and not re.fullmatch(
                                    rect.settings["format_regex"], text
                                )
                            ):
                                textstate = TextDetectionTargetWithResult.ResultState.FailedFilter
                                smoother.clear()
                        else:
                            smoother.clear()
                if "remove_leading_zeros" in rect.settings:
                    if rect.settings["remove_leading_zeros"]:
                        # remove leading zeros
                        text = text.lstrip("0")
                        if text == "":
                            text = "0"
                if "ordinal_indicator" in rect.settings:
                    if rect.settings["ordinal_indicator"]:
                        # add ordinal indicator
                        text = add_ordinal_indicator(text)

            if text == "":
                textstate = TextDetectionTargetWithResult.ResultState.Empty

            result = TextDetectionResult(text, textstate, effectiveRect, extras)

            texts.append(result)
        return texts
