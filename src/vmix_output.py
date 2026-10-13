import threading
from urllib.parse import urlencode

import requests

from text_detection_target import TextDetectionTargetWithResult
from sc_logging import logger
from storage import subscribe_to_data, fetch_data


class VMixAPI:
    REQUEST_TIMEOUT = (1, 2)

    def __init__(self, host, port, input_number, field_mapping):
        self.host = host
        self.port = port
        self.input_number = input_number
        self.field_mapping = dict(field_mapping)
        self._lock = threading.Lock()
        self._running = False
        self._generation = 0
        self._pending = {}
        self._dirty = {}
        self._worker = None
        self.update_same = fetch_data("scoresight.json", "vmix_send_same", False)
        subscribe_to_data("scoresight.json", "vmix_send_same", self.set_update_same)

    @property
    def running(self):
        with self._lock:
            return self._running

    @running.setter
    def running(self, value):
        with self._lock:
            self._running = bool(value)
            if not value:
                self._generation += 1
                self._pending.clear()
                self._dirty.clear()

    def set_update_same(self, update_same):
        self.update_same = update_same

    def set_connection(self, host, port, input_number):
        with self._lock:
            self.host, self.port, self.input_number = host, port, input_number
            self._generation += 1
            self._pending.clear()
            self._dirty.clear()

    def set_field_mapping(self, field_mapping):
        with self._lock:
            self.field_mapping = dict(field_mapping)
            self._generation += 1
            self._pending.clear()
            self._dirty.clear()

    def update_vmix(self, detection: list[TextDetectionTargetWithResult]):
        # HTTP must never hold up the Qt result slot. Keep only the newest
        # pending value per title field when vMix is slower than OCR.
        with self._lock:
            if not self._running or not self.field_mapping:
                return
            # Retry failed fields on the next OCR update, never in a tight loop.
            for field, value in self._dirty.items():
                self._pending.setdefault(field, value)
            states = [TextDetectionTargetWithResult.ResultState.Success]
            if self.update_same:
                states.append(TextDetectionTargetWithResult.ResultState.SameNoChange)
            for target in detection:
                field = self.field_mapping.get(target.name)
                if not field:
                    continue
                if (
                    target.result_state
                    == TextDetectionTargetWithResult.ResultState.Empty
                    and not (target.settings or {}).get("skip_empty", False)
                ):
                    self._pending[field] = ""
                elif target.result_state in states or (
                    target.result_state
                    == TextDetectionTargetWithResult.ResultState.SameNoChange
                    and field in self._dirty
                ):
                    self._pending[field] = target.result or ""
            if self._pending and self._worker is None:
                self._worker = threading.Thread(target=self._send_pending, daemon=True)
                self._worker.start()

    def _send_pending(self):
        while True:
            with self._lock:
                if not self._running or not self._pending:
                    self._worker = None
                    return
                # Take one field at a time so a newer value can replace another
                # field while this request is in flight.
                key = next(iter(self._pending))
                value = self._pending.pop(key)
                # The in-flight value supersedes any failed older value. Do not
                # let another OCR update requeue that stale retry during HTTP.
                self._dirty.pop(key, None)
                generation = self._generation
                query = {
                    "Function": "SetText",
                    "Input": self.input_number,
                    "SelectedName": key,
                    "Value": value,
                }
                url = f"http://{self.host}:{self.port}/api/?{urlencode(query)}"
            with self._lock:
                if not self._running or generation != self._generation:
                    continue
            succeeded = False
            try:
                response = requests.get(url, timeout=self.REQUEST_TIMEOUT)
                succeeded = response.status_code == 200
                if not succeeded:
                    logger.error(
                        f"Failed to send vMix data, status code: {response.status_code}"
                    )
            except requests.exceptions.RequestException as exc:
                logger.error(f"Failed to send data to vMix: {exc}")
            with self._lock:
                if self._running and generation == self._generation:
                    if succeeded:
                        self._dirty.pop(key, None)
                    else:
                        self._dirty[key] = value
