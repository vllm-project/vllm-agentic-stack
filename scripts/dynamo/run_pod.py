"""Run a bounded single-host Dynamo recording session and retain diagnostics.

Requires prepare-pod.sh. No Docker daemon or cloud API access is used.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[2]
MODEL = "openai/gpt-oss-20b"


def wait_for_model(url, timeout):
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=5) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(f"{url}/v1/models/{quote(MODEL, safe='')}/ready",
                                      timeout=min(5, max(0.01, deadline - time.monotonic())))
                if response.status_code == 200 and response.json().get("ready") is True:
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise RuntimeError("model readiness deadline exceeded; inspect frontend.log and worker.log")


def smoke(url):
    # A client-executed function: no search subscription or implicit gateway tool.
    body = {"model": MODEL, "stream": False, "max_tokens": 2048,
            "messages": [{"role": "user", "content": "Use lookup_tag to get the tag, then repeat it."}],
            "tools": [{"name": "lookup_tag", "description": "Return the test tag.",
                       "input_schema": {"type": "object", "properties": {}}}]}
    with httpx.Client(timeout=120) as client:
        first = client.post(f"{url}/v1/messages", json=body)
        first.raise_for_status()
        message = first.json()
        calls = [b for b in message["content"] if b["type"] == "tool_use"]
        if len(calls) != 1 or calls[0]["name"] != "lookup_tag" or message["stop_reason"] != "tool_use":
            raise RuntimeError("gateway smoke did not produce the expected client tool call")
        body["messages"] += [{"role": "assistant", "content": message["content"]},
                             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": calls[0]["id"],
                                                             "content": "DYNAMO_SMOKE_OK"}]}]
        final = client.post(f"{url}/v1/messages", json=body)
        final.raise_for_status()
        result = final.json()
        if result.get("stop_reason") != "end_turn" or not any(
            b["type"] == "text" and "DYNAMO_SMOKE_OK" in b.get("text", "") for b in result["content"]
        ):
            raise RuntimeError("gateway smoke did not finish with the tool output")
        return {"first": message, "final": result}



def stop_children(children):
    """Stop only process groups created by this runner and reap their leaders."""
    for child in reversed(children):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "target/dynamo-session")
    parser.add_argument("--ready-timeout", type=int, default=900)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("GPU recording requires a Linux x86_64 Pod; local tests run separately")
    if args.ready_timeout <= 0:
        parser.error("--ready-timeout must be positive")
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        parser.error("need at least 30 GiB free after preparation")
    for port in (8000, 9000, 7070):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    # Exclusive session output prevents accidental overwrite or concurrent sessions.
    args.output.mkdir(parents=True, exist_ok=False)
    dynamo_python = ROOT / ".venv-dynamo/bin/python"
    commands = {
        "frontend": [str(dynamo_python), "-m", "dynamo.frontend", "--http-port", "8000",
                     "--discovery-backend", "file", "--enable-anthropic-api"],
        "worker": [str(dynamo_python), "-m", "dynamo.vllm", "--model", MODEL,
                   "--discovery-backend", "file", "--kv-events-config", '{"enable_kv_cache_events":false}',
                   "--dyn-reasoning-parser", "gpt_oss", "--dyn-tool-call-parser", "harmony",
                   "--max-model-len", "8192", "--max-num-seqs", "1"],
        "gateway": [str(ROOT / "target/debug/agentic-server"), "--llm-api-base", "http://127.0.0.1:8000"],
    }
    metadata = {"model": MODEL, "commands": commands,
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
                "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                                "--format=csv"], text=True)}
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    for name, executable in (("dynamo", dynamo_python), ("recorder", Path(sys.executable))):
        result = subprocess.check_output([str(executable), "-c",
            "import importlib.metadata as m; print('\\n'.join(sorted(f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions())))"], text=True)
        (args.output / f"{name}-packages.txt").write_text(result)
    children = []
    logs = []

    def start(name):
        log = (args.output / f"{name}.log").open("w")
        logs.append(log)
        child = subprocess.Popen(commands[name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)

    def run_owned(command, log, timeout, env=None):
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        status = child.wait(timeout=timeout)
        if status:
            raise subprocess.CalledProcessError(status, command)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        start("frontend")
        start("worker")
        wait_for_model("http://127.0.0.1:8000", args.ready_timeout)
        # Record exact downloaded revision rather than pretending the mutable name is pinned.
        code = "from huggingface_hub import snapshot_download; print(snapshot_download('openai/gpt-oss-20b', local_files_only=True))"
        snapshot = subprocess.check_output([str(dynamo_python), "-c", code], text=True).strip()
        metadata["model_snapshot"] = snapshot
        (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
        env = dict(os.environ, PYTHON=sys.executable, MODEL=MODEL, DYNAMO_URL="http://127.0.0.1:8000")
        with (args.output / "recording.log").open("w") as log:
            run_owned(["bash", "crates/agentic-server-core/tests/cassettes/record_dynamo_messages_cassettes.sh"],
                      log, 900, env)
        start("gateway")
        deadline = time.monotonic() + 30
        with httpx.Client(timeout=2) as client:
            while True:
                try:
                    if client.get("http://127.0.0.1:9000/health").is_success:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("gateway readiness deadline exceeded")
                time.sleep(0.2)
        (args.output / "gateway-smoke.json").write_text(json.dumps(smoke("http://127.0.0.1:9000"), indent=2))
        with (args.output / "replay.log").open("w") as log:
            run_owned(["cargo", "test", "--locked", "-p", "agentic-server-core", "--test", "dynamo_messages_test",
                       "dynamo_messages_recorded_acceptance", "--", "--ignored", "--exact"], log, 600)
        for path in (ROOT / "crates/agentic-server-core/tests/cassettes/dynamo").glob("dynamo-messages-*.yaml"):
            shutil.copy2(path, args.output)
        (args.output / "SUCCESS").write_text("Live recording, gateway smoke, and offline replay passed.\n")
    finally:
        signal.signal(signal.SIGTERM, previous)
        stop_children(children)
        for log in logs:
            log.close()
        print(f"Session artifacts: {args.output}")


if __name__ == "__main__":
    main()
