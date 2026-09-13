"""Exercise deployment control behavior locally; does not claim CUDA compatibility."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pod


class PodTests(unittest.TestCase):
    def test_model_readiness_and_encoded_model_id(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"ready": True})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.object(run_pod.httpx, "Client", return_value=client):
            run_pod.wait_for_model("http://localhost:8000", 1)
        self.assertIn(b"openai%2Fgpt-oss-20b", requests[0].url.raw_path)

    def test_model_readiness_failure_has_deadline(self):
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
        with patch.object(run_pod.httpx, "Client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "readiness deadline"):
                run_pod.wait_for_model("http://localhost:8000", 0.01)

    def test_smoke_preserves_tool_call_and_result(self):
        import json
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                return httpx.Response(200, json={"stop_reason": "tool_use", "content": [
                    {"type": "tool_use", "id": "call_test", "name": "lookup_tag", "input": {}}]})
            return httpx.Response(200, json={"stop_reason": "end_turn", "content": [
                {"type": "text", "text": "DYNAMO_SMOKE_OK"}]})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.object(run_pod.httpx, "Client", return_value=client):
            run_pod.smoke("http://localhost:9000")
        self.assertEqual(requests[1]["messages"][2]["content"][0]["tool_use_id"], "call_test")
        self.assertEqual(requests[1]["messages"][1]["content"][0]["name"], "lookup_tag")

    def test_smoke_rejects_no_tool(self):
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
            200, json={"stop_reason": "end_turn", "content": []})))
        with patch.object(run_pod.httpx, "Client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "expected client tool"):
                run_pod.smoke("http://localhost:9000")

    @unittest.skipUnless(os.name == "posix", "Pod and MacBook use POSIX process groups")
    def test_cleanup_reaps_only_owned_process(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            run_pod.stop_children([child])
            self.assertIsNotNone(child.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            run_pod.stop_children([child, unrelated])


if __name__ == "__main__":
    unittest.main()
