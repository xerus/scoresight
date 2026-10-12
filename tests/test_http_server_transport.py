"""Exercise the JSON endpoint over a real HTTP connection on an ephemeral port."""

import http.client
import json
import socket
import threading
import time
import unittest

import uvicorn

import http_server
from text_detection_target import TextDetectionTarget, TextDetectionTargetWithResult


class JsonTransportTests(unittest.TestCase):
    def test_uvicorn_serves_current_results_and_shuts_down_cleanly(self):
        previous = http_server.http_results
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen(16)
        config = uvicorn.Config(
            http_server.app,
            host="127.0.0.1",
            port=port,
            loop="asyncio",
            log_level="error",
            access_log=False,
            lifespan="on",
            timeout_graceful_shutdown=1,
        )
        server = uvicorn.Server(config)
        failures = []

        def serve():
            try:
                server.run(sockets=[listener])
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=serve, daemon=True)
        connection = None
        try:
            http_server.update_http_server([])
            thread.start()
            deadline = time.monotonic() + 5
            while (
                not server.started and thread.is_alive() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertTrue(server.started, f"Server did not start: {failures}")
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            connection.request("GET", "/json")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read()), [])

            target = TextDetectionTarget(1, 2, 30, 40, "Domácí skóre", {})
            result = TextDetectionTargetWithResult(
                target, "0", TextDetectionTargetWithResult.ResultState.Success
            )
            http_server.update_http_server([result])
            connection.request("GET", "/json?pivot")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Type"), "application/json")
            self.assertEqual(json.loads(response.read()), {"Domácí skóre": "0"})

            result.result = "17"
            http_server.update_http_server([result])
            connection.request("GET", "/json")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())[0]["text"], "17")
        finally:
            if connection is not None:
                connection.close()
            server.should_exit = True
            if thread.ident is not None:
                thread.join(timeout=5)
            listener.close()
            http_server.update_http_server(previous)
            self.assertFalse(thread.is_alive(), "Uvicorn test thread did not stop")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
