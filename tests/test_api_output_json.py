"""Outgoing JSON keeps accepted text through gaps before starting send workers."""

import json
import unittest
from unittest.mock import patch

import api_output
from json_output import JsonOutput
from text_detection_target import TextDetectionTarget, TextDetectionTargetWithResult


class ApiOutputJsonTests(unittest.TestCase):
    def test_post_and_put_hold_text_for_full_and_simple_json(self):
        states = TextDetectionTargetWithResult.ResultState
        for method in ("POST", "PUT"):
            for encoding in ("JSON (Full)", "JSON (Simple key-value)"):
                with (
                    self.subTest(method=method, encoding=encoding),
                    patch.multiple(
                        api_output,
                        out_api_url="https://example.test/score",
                        out_api_method=method,
                        out_api_encoding=encoding,
                        json_output=JsonOutput(),
                    ),
                    patch("api_output.threading.Thread") as thread,
                    patch("api_output.requests." + method.lower()) as send,
                ):
                    thread.side_effect = lambda target: type(
                        "ImmediateWorker", (), {"start": staticmethod(target)}
                    )()
                    send.return_value.status_code = 200
                    for text, state, expected in (
                        ("17", states.Success, "17"),
                        ("", states.Empty, "17"),
                        ("71", states.FailedFilter, "17"),
                        ("0", states.Success, "0"),
                        (None, states.FailedFilter, "0"),
                    ):
                        target = TextDetectionTarget(0, 0, 50, 50, "Away", {})
                        raw = TextDetectionTargetWithResult(target, text, state)
                        api_output.update_out_api([raw])
                        payload = json.loads(send.call_args.kwargs["data"])
                        if encoding == "JSON (Full)":
                            self.assertEqual(payload[0]["text"], expected)
                            self.assertEqual(payload[0]["state"], state.name)
                        else:
                            self.assertEqual(payload, {"Away": expected})
                        self.assertEqual(raw.result, text)
