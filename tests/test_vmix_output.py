"""vMix API contract and GUI mapping regressions; no vMix installation needed."""

import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import requests
import shiboken6
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QTableView

from text_detection_target import TextDetectionTarget, TextDetectionTargetWithResult
from vmix_output import VMixAPI
from vmix_ui_handler import VMixUIHanlder

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

State = TextDetectionTargetWithResult.ResultState


def result(name="Home", text="7", state=State.Success, settings=None):
    return TextDetectionTargetWithResult(
        TextDetectionTarget(0, 0, 10, 10, name, settings or {}), text, state
    )


class VMixOutputTests(unittest.TestCase):
    def setUp(self):
        self.fetch = patch("vmix_output.fetch_data", return_value=False)
        self.subscribe = patch("vmix_output.subscribe_to_data")
        self.fetch.start()
        self.subscribe.start()
        self.addCleanup(self.fetch.stop)
        self.addCleanup(self.subscribe.stop)
        self.api = VMixAPI("localhost", "8088", "1", {"Home": "Home.Text"})
        self.addCleanup(self.stop_worker)
        self.api.running = True

    def stop_worker(self):
        worker = self.api._worker
        self.api.running = False
        if worker:
            worker.join(5)
            self.assertFalse(worker.is_alive(), "vMix worker did not finish")

    def wait_for_updates(self):
        worker = self.api._worker
        if worker:
            worker.join(5)
            self.assertFalse(worker.is_alive(), "vMix worker did not finish")

    def test_set_text_contract_unicode_reserved_characters_and_gt_field(self):
        self.api.set_connection("localhost", "8088", "Score & time.gtzip")
        self.api.set_field_mapping({"Home": "Skóre + Home.Text"})
        with patch(
            "vmix_output.requests.get", return_value=SimpleNamespace(status_code=200)
        ) as get:
            self.api.update_vmix([result(text="Žluťoučký & + = ? #\n0")])
            self.wait_for_updates()
        url = urlsplit(get.call_args.args[0])
        self.assertEqual(
            (url.scheme, url.netloc, url.path), ("http", "localhost:8088", "/api/")
        )
        self.assertEqual(
            parse_qs(url.query, keep_blank_values=True),
            {
                "Function": ["SetText"],
                "Input": ["Score & time.gtzip"],
                "SelectedName": ["Skóre + Home.Text"],
                "Value": ["Žluťoučký & + = ? #\n0"],
            },
        )
        self.assertEqual(get.call_args.kwargs, {"timeout": (1, 2)})

    def test_real_http_request_updates_mock_title_server(self):
        received = []

        class TitleServer(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(3)

            def do_GET(self):
                received.append(
                    parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                )
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), TitleServer)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            self.api.set_connection("127.0.0.1", str(server.server_port), "1")
            self.api.update_vmix([result(text="7 & 1")])
            self.wait_for_updates()
        finally:
            server.shutdown()
            server.server_close()
            worker.join(5)
        self.assertEqual(
            received,
            [
                {
                    "Function": ["SetText"],
                    "Input": ["1"],
                    "SelectedName": ["Home.Text"],
                    "Value": ["7 & 1"],
                }
            ],
        )

    def test_stopped_unmapped_failed_and_unchanged_results_do_not_send(self):
        with patch("vmix_output.requests.get") as get:
            self.api.running = False
            self.api.update_vmix([result()])
            self.api.running = True
            self.api.update_vmix(
                [
                    result(state=State.SameNoChange),
                    result(state=State.FailedFilter),
                    result(name="Away"),
                ]
            )
            self.wait_for_updates()
            get.assert_not_called()

    def test_zero_and_same_option_are_sent(self):
        self.api.set_update_same(True)
        with patch(
            "vmix_output.requests.get", return_value=SimpleNamespace(status_code=200)
        ) as get:
            self.api.update_vmix([result(text="0", state=State.SameNoChange)])
            self.wait_for_updates()
            self.assertEqual(
                parse_qs(urlsplit(get.call_args.args[0]).query)["Value"], ["0"]
            )

    def test_confirmed_blank_clears_title_but_skip_empty_preserves_it(self):
        with patch(
            "vmix_output.requests.get", return_value=SimpleNamespace(status_code=200)
        ) as get:
            self.api.update_vmix([result(text="ignored", state=State.Empty)])
            self.wait_for_updates()
            self.assertEqual(
                parse_qs(urlsplit(get.call_args.args[0]).query, keep_blank_values=True)[
                    "Value"
                ],
                [""],
            )
            get.reset_mock()
            self.api.update_vmix(
                [result(state=State.Empty, settings={"skip_empty": True})]
            )
            self.wait_for_updates()
            get.assert_not_called()

    def test_empty_mapping_disables_field(self):
        self.api.set_field_mapping({"Home": ""})
        with patch("vmix_output.requests.get") as get:
            self.api.update_vmix([result()])
            self.wait_for_updates()
            get.assert_not_called()

    def test_http_failure_and_timeout_are_logged_and_other_fields_continue(self):
        self.api.set_field_mapping(
            {"Home": "Home.Text", "Away": "Away.Text", "Clock": "Clock.Text"}
        )
        with (
            patch(
                "vmix_output.requests.get",
                side_effect=[
                    SimpleNamespace(status_code=500),
                    requests.Timeout("timed out"),
                    SimpleNamespace(status_code=200),
                ],
            ) as get,
            patch("vmix_output.logger.error") as error,
        ):
            self.api.update_vmix([result(), result(name="Away"), result(name="Clock")])
            self.wait_for_updates()
            self.assertEqual(get.call_count, 3)
            self.assertEqual(error.call_count, 2)

    def test_failed_field_retries_on_unchanged_update_until_success_then_stops(self):
        with (
            patch(
                "vmix_output.requests.get",
                side_effect=[
                    requests.Timeout("offline"),
                    SimpleNamespace(status_code=200),
                ],
            ) as get,
            patch("vmix_output.logger.error"),
        ):
            self.api.update_vmix([result(text="7")])
            self.wait_for_updates()
            self.assertEqual(get.call_count, 1)
            self.assertIsNone(self.api._worker)
            self.api.update_vmix([result(text="7", state=State.SameNoChange)])
            self.wait_for_updates()
            self.assertEqual(get.call_count, 2)
            self.api.update_vmix([result(text="7", state=State.SameNoChange)])
            self.wait_for_updates()
            self.assertEqual(get.call_count, 2)

    def test_newer_inflight_value_cannot_be_rolled_back_by_stale_retry(self):
        entered, release = threading.Event(), threading.Event()
        values = []

        def send(url, **kwargs):
            values.append(parse_qs(urlsplit(url).query)["Value"][0])
            if len(values) == 1:
                raise requests.Timeout("first value failed")
            if len(values) == 2:
                entered.set()
                release.wait(5)
            return SimpleNamespace(status_code=200)

        with (
            patch("vmix_output.requests.get", side_effect=send),
            patch("vmix_output.logger.error"),
        ):
            try:
                self.api.update_vmix([result(text="41")])
                self.wait_for_updates()
                self.api.update_vmix([result(text="42")])
                self.assertTrue(entered.wait(2))
                self.api.update_vmix([])
                self.api.update_vmix([result(text="bad", state=State.FailedFilter)])
            finally:
                release.set()
                self.wait_for_updates()
        self.assertEqual(values, ["41", "42"])
        self.assertEqual(self.api._dirty, {})

    def test_slow_network_does_not_block_and_pending_values_are_coalesced(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def send(url, **kwargs):
            calls.append(parse_qs(urlsplit(url).query)["Value"][0])
            if len(calls) == 1:
                entered.set()
                if not release.wait(5):
                    raise requests.Timeout("test worker was not released")
            return SimpleNamespace(status_code=200)

        with patch("vmix_output.requests.get", side_effect=send):
            try:
                self.api.update_vmix([result(text="1")])
                self.assertTrue(entered.wait(2))
                self.api.update_vmix([result(text="2")])
                self.api.update_vmix([result(text="3")])
                self.assertEqual(calls, ["1"])
            finally:
                release.set()
                self.wait_for_updates()
        self.assertEqual(calls, ["1", "3"])

    def test_stop_discards_pending_requests(self):
        entered, release = threading.Event(), threading.Event()

        def send(*args, **kwargs):
            entered.set()
            release.wait(5)
            return SimpleNamespace(status_code=200)

        with patch("vmix_output.requests.get", side_effect=send) as get:
            try:
                self.api.update_vmix([result(text="1")])
                self.assertTrue(entered.wait(2))
                self.api.update_vmix([result(text="2")])
                self.api.running = False
            finally:
                release.set()
                self.wait_for_updates()
            self.assertEqual(get.call_count, 1)

    def test_connection_change_discards_old_pending_but_preserves_mapping_and_running(
        self,
    ):
        entered, release = threading.Event(), threading.Event()

        def send(*args, **kwargs):
            entered.set()
            release.wait(5)
            return SimpleNamespace(status_code=200)

        with patch("vmix_output.requests.get", side_effect=send) as get:
            try:
                self.api.update_vmix([result(text="1")])
                self.assertTrue(entered.wait(2))
                self.api.update_vmix([result(text="2")])
                self.api.set_connection("newhost", "9000", "title-name")
                self.api.update_vmix([result(text="3")])
            finally:
                release.set()
                self.wait_for_updates()
            self.assertEqual(get.call_count, 2)
            url = urlsplit(get.call_args.args[0])
            self.assertEqual(url.netloc, "newhost:9000")
            self.assertEqual(parse_qs(url.query)["Value"], ["3"])
        self.assertTrue(self.api.running)
        self.assertEqual(self.api.field_mapping, {"Home": "Home.Text"})


class VMixMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.settings = {}

        def fetch(file, key, default=None):
            return self.settings.get(key, default)

        for module in ("vmix_ui_handler", "vmix_output"):
            patcher = patch(f"{module}.fetch_data", side_effect=fetch)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("vmix_output.subscribe_to_data")
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch(
            "vmix_ui_handler.store_data",
            side_effect=lambda file, key, value: self.settings.update({key: value}),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ui = SimpleNamespace(
            lineEdit_vmixHost=QLineEdit(),
            lineEdit_vmixPort=QLineEdit(),
            inputLineEdit_vmix=QLineEdit(),
            tableView_vmixMapping=QTableView(),
            pushButton_startvmix=QPushButton(),
        )
        self.ui.pushButton_startvmix.setCheckable(True)
        self.handler = VMixUIHanlder(self.ui)

    def tearDown(self):
        # Qt signal connections keep the handler and widgets in a reference
        # cycle. Destroy widgets on the main thread, before later HTTP workers
        # can trigger garbage collection and deferred Shiboken deletion.
        for widget in vars(self.ui).values():
            shiboken6.delete(widget)
        self.handler.ui = None
        self.ui = None

    def test_default_mapping_applied_without_edit_then_edit_and_remove(self):
        home = TextDetectionTarget(0, 0, 10, 10, "Home", {})
        self.handler.updatevMixTable([home])
        self.assertEqual(self.handler.vmixUpdater.field_mapping, {"Home": "Home"})
        self.ui.tableView_vmixMapping.model().item(0, 1).setText("Home.Text")
        self.assertEqual(self.handler.vmixUpdater.field_mapping, {"Home": "Home.Text"})
        self.handler.updatevMixTable([home])
        self.assertEqual(self.handler.vmixUpdater.field_mapping, {"Home": "Home.Text"})
        self.handler.updatevMixTable([])
        self.assertEqual(self.handler.vmixUpdater.field_mapping, {})

    def test_connection_edit_preserves_running_mapping_and_single_updater(self):
        self.assertEqual(self.ui.lineEdit_vmixPort.text(), "8088")
        updater = self.handler.vmixUpdater
        updater.set_field_mapping({"Home": "Home.Text"})
        self.ui.pushButton_startvmix.setChecked(True)
        self.ui.lineEdit_vmixHost.setText("192.0.2.10")
        self.assertIs(self.handler.vmixUpdater, updater)
        self.assertTrue(updater.running)
        self.assertEqual(updater.field_mapping, {"Home": "Home.Text"})
        self.assertEqual(self.settings["vmix_host"], "192.0.2.10")
        self.ui.pushButton_startvmix.setChecked(False)
        self.assertFalse(updater.running)


if __name__ == "__main__":
    unittest.main()
