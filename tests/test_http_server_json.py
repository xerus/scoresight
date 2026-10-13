"""Exercise the JSON API through ASGI without binding a port or starting Qt."""

import asyncio
import json
import threading
import unittest
from unittest.mock import patch

import http_server
from text_detection_target import TextDetectionTarget, TextDetectionTargetWithResult


class HttpServerJsonTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.previous_results = http_server.http_results
        http_server.update_http_server([])

    def tearDown(self):
        http_server.update_http_server(self.previous_results)

    def result(self, name, text, state=None, settings=None):
        target = TextDetectionTarget(1.5, 2.5, 80, 40, name, settings or {})
        return TextDetectionTargetWithResult(
            target,
            text,
            state or TextDetectionTargetWithResult.ResultState.Success,
        )

    async def request(self, path="/json", query=b"", method="GET", headers=()):
        """Collect the actual status, headers and serialized response body."""
        messages = []
        request_received = False

        async def receive():
            nonlocal request_received
            if not request_received:
                request_received = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "root_path": "",
            "query_string": query,
            "headers": list(headers),
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }
        await asyncio.wait_for(http_server.app(scope, receive, send), timeout=3)
        start = next(m for m in messages if m["type"] == "http.response.start")
        body = b"".join(
            m.get("body", b"") for m in messages if m["type"] == "http.response.body"
        )
        return start["status"], dict(start["headers"]), body

    async def json_response(self, **kwargs):
        status, headers, body = await self.request(**kwargs)
        self.assertEqual(status, 200)
        self.assertEqual(headers[b"content-type"], b"application/json")
        self.assertEqual(int(headers[b"content-length"]), len(body))
        return json.loads(body)

    async def test_empty_results_have_list_and_pivot_shapes(self):
        self.assertEqual(await self.json_response(), [])
        self.assertEqual(await self.json_response(query=b"pivot"), {})

    async def test_default_response_includes_geometry_and_all_result_states(self):
        state = TextDetectionTargetWithResult.ResultState
        http_server.update_http_server(
            [
                self.result("Home Score", "0"),
                self.result("Time", "01:07", state.SameNoChange),
                self.result("Blank", "", state.Empty, {"skip_empty": True}),
                self.result("Uncertain", None, state.FailedFilter),
            ]
        )
        data = await self.json_response()
        self.assertEqual(
            [(item["name"], item["text"], item["state"]) for item in data],
            [
                ("Home Score", "0", "Success"),
                ("Time", "01:07", "SameNoChange"),
                ("Blank", "", "Empty"),
                ("Uncertain", None, "FailedFilter"),
            ],
        )
        self.assertEqual(set(data[0]), {"name", "text", "state", "rect"})
        self.assertEqual(
            data[0]["rect"], {"x": 1.5, "y": 2.5, "width": 80.0, "height": 40.0}
        )

    async def test_pivot_distinguishes_zero_blank_skipped_blank_and_unknown(self):
        state = TextDetectionTargetWithResult.ResultState
        http_server.update_http_server(
            [
                self.result("Zero", "0", settings={"skip_empty": True}),
                self.result("Unchanged", "7", state.SameNoChange),
                self.result("Blank", "", state.Empty),
                self.result("Allowed blank", "", state.Empty, {"skip_empty": False}),
                self.result("Skipped blank", "", state.Empty, {"skip_empty": True}),
                self.result("Unknown", None, state.FailedFilter),
                self.result("Rejected text", "31", state.FailedFilter),
            ]
        )
        self.assertEqual(
            await self.json_response(query=b"pivot"),
            {"Zero": "0", "Unchanged": "7", "Blank": "", "Allowed blank": ""},
        )

    async def test_pivot_accepts_missing_settings(self):
        result = self.result(
            "Blank", "", TextDetectionTargetWithResult.ResultState.Empty
        )
        result.settings = None
        http_server.update_http_server([result])
        self.assertEqual(await self.json_response(query=b"pivot"), {"Blank": ""})

    async def test_pivot_is_enabled_by_presence_of_query_parameter(self):
        http_server.update_http_server([self.result("Home Score", "17")])
        for query in (b"pivot", b"pivot=", b"pivot=true", b"pivot=false", b"pivot=0"):
            with self.subTest(query=query):
                self.assertEqual(
                    await self.json_response(query=query), {"Home Score": "17"}
                )
        self.assertIsInstance(await self.json_response(query=b"other=pivot"), list)

    async def test_pivot_duplicate_names_use_last_accepted_value(self):
        http_server.update_http_server(
            [
                self.result("Score", "1"),
                self.result("Score", "7"),
                self.result(
                    "Score",
                    "31",
                    TextDetectionTargetWithResult.ResultState.FailedFilter,
                ),
            ]
        )
        self.assertEqual(await self.json_response(query=b"pivot"), {"Score": "7"})
        self.assertEqual(len(await self.json_response()), 3)

    async def test_unicode_and_special_characters_round_trip(self):
        name = 'Čas 🏀 "domácí"'
        text = 'Žluťoučký\\\n\t"7"'
        http_server.update_http_server([self.result(name, text)])
        status, _, body = await self.request()
        self.assertEqual(status, 200)
        self.assertIn("Čas 🏀".encode("utf-8"), body)
        self.assertEqual(json.loads(body)[0]["text"], text)
        self.assertEqual(await self.json_response(query=b"pivot"), {name: text})

    async def test_updates_replace_previous_fields_and_clear_results(self):
        http_server.update_http_server([self.result("Old", "1")])
        self.assertEqual(await self.json_response(query=b"pivot"), {"Old": "1"})
        http_server.update_http_server([self.result("New", "7")])
        self.assertEqual(await self.json_response(query=b"pivot"), {"New": "7"})
        http_server.update_http_server([])
        self.assertEqual(await self.json_response(), [])
        self.assertEqual(await self.json_response(query=b"pivot"), {})

    async def test_worker_update_during_serialization_preserves_current_batch(self):
        first = self.result("First", "1")
        second = self.result("Second", "7")
        replacement = self.result("Next frame", "17")
        http_server.update_http_server([first, second])
        serialize = first.to_dict

        def serialize_and_update():
            worker = threading.Thread(
                target=http_server.update_http_server, args=([replacement],)
            )
            worker.start()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            return serialize()

        with patch.object(first, "to_dict", side_effect=serialize_and_update):
            data = await self.json_response()
        self.assertEqual([item["name"] for item in data], ["First", "Second"])
        self.assertEqual(await self.json_response(query=b"pivot"), {"Next frame": "17"})

    async def test_cors_allows_browser_get_requests(self):
        status, headers, body = await self.request(
            headers=[(b"origin", b"https://overlay.example")]
        )
        self.assertEqual(status, 200)
        self.assertIn(
            headers[b"access-control-allow-origin"], (b"*", b"https://overlay.example")
        )
        self.assertEqual(json.loads(body), [])

    async def test_cors_preflight_allows_requested_headers(self):
        status, headers, _ = await self.request(
            method="OPTIONS",
            headers=[
                (b"origin", b"https://overlay.example"),
                (b"access-control-request-method", b"GET"),
                (b"access-control-request-headers", b"X-Overlay"),
            ],
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            headers[b"access-control-allow-origin"], b"https://overlay.example"
        )
        self.assertIn(b"GET", headers[b"access-control-allow-methods"])
        self.assertEqual(headers[b"access-control-allow-headers"], b"X-Overlay")

    async def test_post_is_rejected_and_unknown_route_is_not_found(self):
        status, headers, body = await self.request(method="POST")
        self.assertEqual(status, 405)
        self.assertIn(b"GET", headers[b"allow"])
        self.assertEqual(json.loads(body), {"detail": "Method Not Allowed"})
        status, _, body = await self.request(path="/json/missing")
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body), {"detail": "Not Found"})


if __name__ == "__main__":
    unittest.main()
