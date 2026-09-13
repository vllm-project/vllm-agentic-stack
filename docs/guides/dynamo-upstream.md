# Running Agentic API in front of NVIDIA Dynamo

[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) is a distributed inference serving framework. Its frontend
exposes an OpenAI-compatible HTTP API, including `/v1/responses`, and routes requests to backend workers. Dynamo ships
its own workers (vLLM, SGLang, TensorRT-LLM); the vLLM worker (`python -m dynamo.vllm`) embeds the vLLM engine, so a
Dynamo deployment is a complete inference stack on its own. You do not run `vllm serve` alongside it.

This guide records how to put Agentic API in front of a Dynamo deployment and what differs from pointing the gateway
at a standalone `vllm serve`.

The short version: nothing in Agentic API needs to change. Start the gateway with `--llm-api-base` pointing at the
Dynamo frontend and it works, including stateful `previous_response_id` chaining and client-executed function tools.

## What Dynamo does and does not provide

| Capability | Dynamo frontend | Agentic API adds |
|---|---|---|
| `POST /v1/responses`, `/v1/chat/completions`, `/v1/models`, `/health` | ✅ | — |
| Reasoning / tool-call parsing | ✅ via `--dyn-reasoning-parser` / `--dyn-tool-call-parser` on the worker | — |
| `previous_response_id` | ❌ returns `501 Not Implemented` (`Validation: previous_response_id is not supported.`) | ✅ Stores every response and rehydrates the full item history on each turn, so the upstream call is stateless |
| Gateway-executed built-in tools (web search, MCP), background execution, WebSocket transport | ❌ | ✅ |

Because Dynamo rejects `previous_response_id`, the gateway never forwards it. The second turn of a conversation reaches
Dynamo as one `input` array containing the earlier user message, the stored assistant message, and the new user
message. The replay tests in `crates/agentic-server-core/tests/dynamo_cassette_test.rs` assert exactly that shape.

## 1. Install Dynamo

Dynamo publishes wheels on PyPI. The `[vllm]` extra pulls in the vLLM version Dynamo's worker is built against, so use
a dedicated virtual environment, and pin the Dynamo release:

```bash
mkdir -p ~/dev/dynamo && cd ~/dev/dynamo
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install "ai-dynamo[vllm]==1.4.1"
```

Pin the version. Dynamo's README suggests `uv pip install --prerelease=allow "ai-dynamo[vllm]"`, but that flag lets uv
resolve *any* dependency to a pre-release and, with an unpinned `ai-dynamo`, installs the latest `1.5.0.devYYYYMMDD`
build rather than a release. Check what you got with `python -c 'import importlib.metadata as m; print(m.version("ai-dynamo"))'`.

This guide was verified with `ai-dynamo==1.4.1` (which installs `vllm==0.26.0` and `torch` cu130) on an aarch64 host
with a single GB10 GPU. See Dynamo's [release artifacts](https://docs.nvidia.com/dynamo/resources/release-artifacts)
and [support matrix](https://docs.nvidia.com/dynamo/resources/support-matrix) for the wheel/CUDA combinations of other
releases. No etcd or NATS is needed for a single-host setup when the components use file-based discovery.

## 2. Start the Dynamo frontend and a worker

Run each in its own terminal (or tmux window). The frontend is the HTTP entry point (default port 8000, round-robin
routing across whatever workers register); the worker loads the model. Models are resolved from the Hugging Face
cache, so anything already downloaded for vLLM is reused.

```bash
# Frontend: OpenAI-compatible HTTP on :8000
.venv/bin/python -m dynamo.frontend --http-port 8000 --discovery-backend file

# Worker: vLLM engine managed by Dynamo
.venv/bin/python -m dynamo.vllm \
  --model openai/gpt-oss-20b \
  --discovery-backend file \
  --kv-events-config '{"enable_kv_cache_events": false}' \
  --dyn-reasoning-parser gpt_oss \
  --dyn-tool-call-parser harmony \
  --max-model-len 32768
```

`openai/gpt-oss-20b` needs roughly 16 GB of GPU memory for weights plus KV cache, so it fits a single 24 GB GPU with
vLLM's default `--gpu-memory-utilization 0.9`. Lower that only when the GPU is shared (see below).

Flags worth knowing:

| Flag | Why |
|---|---|
| `--discovery-backend file` | Lets the frontend and worker find each other via `/tmp/dynamo_store_kv` instead of etcd. Pass it to both. |
| `--kv-events-config '{"enable_kv_cache_events": false}'` | Required for the vLLM worker without NATS. |
| `--dyn-reasoning-parser` / `--dyn-tool-call-parser` | The Dynamo *frontend* parses model output, not vLLM. vLLM's `--reasoning-parser` is ignored and `--tool-call-parser` / `--enable-auto-tool-choice` are rejected as unknown arguments. Without the `--dyn-*` flags, gpt-oss "analysis" text leaks into `content` and tool calls come back as plain text. `gpt_oss` + `harmony` is the verified pair for gpt-oss models; `python -m dynamo.vllm --help` lists the parsers for other model families. |
| `--gpu-memory-utilization` | Standard vLLM engine flag (the worker accepts vLLM engine arguments). It is a fraction of *total* device memory and must fit in the memory currently free, or the engine fails at startup. On a dedicated GPU keep the default; on a shared GPU size it to what is actually free (the recordings for this guide used `0.15` on a 121 GB unified-memory host that was also running another model). |

Confirm the worker registered and parsing works:

```bash
curl -s localhost:8000/v1/models | jq '.data[].id'
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "openai/gpt-oss-20b",
  "messages": [{"role": "user", "content": "Say hello in five words."}],
  "max_tokens": 500
}' | jq '.choices[0].message | {content, reasoning_content}'
```

`content` should hold the answer and `reasoning_content` the chain of thought. If the answer starts with `analysis`, the
worker is missing `--dyn-reasoning-parser`.

## 3. Start Agentic API against the Dynamo frontend

```bash
cargo build -p agentic-server --bins
./target/debug/agentic-server --llm-api-base http://127.0.0.1:8000
```

Agentic API's startup probe uses Dynamo's `/health`, so no `--skip-llm-ready-check` is needed. Be aware of what that
probe means: Dynamo returns `200 {"status":"healthy", ...}` as soon as the frontend's HTTP service is up, even with no
worker registered (the `instances` list is simply empty). It does **not** mean a model is loaded. Wait for the
per-model readiness endpoint before sending traffic:

```bash
curl -s localhost:8000/v1/models/openai%2Fgpt-oss-20b/ready   # 404 "Model not found" until the worker registers,
                                                                # then {"model": "...", "ready": true, ...}
```

The harness CLI works the same way: `./target/debug/agentic run codex --upstream http://127.0.0.1:8000`.

## 4. Verify a stateful conversation and a tool call

```bash
R1=$(curl -s localhost:9000/v1/responses -H 'Content-Type: application/json' -d '{
  "model": "openai/gpt-oss-20b",
  "input": "Remember the word APPLE. Just say: OK",
  "max_output_tokens": 2048
}')
echo "$R1" | jq -r '.output[] | select(.type=="message") | .content[0].text'   # OK

curl -s localhost:9000/v1/responses -H 'Content-Type: application/json' -d "{
  \"model\": \"openai/gpt-oss-20b\",
  \"input\": \"What word did I ask you to remember? Reply with just the word.\",
  \"previous_response_id\": \"$(echo "$R1" | jq -r .id)\",
  \"max_output_tokens\": 2048
}" | jq -r '.output[] | select(.type=="message") | .content[0].text'           # APPLE

curl -s localhost:9000/v1/responses -H 'Content-Type: application/json' -d '{
  "model": "openai/gpt-oss-20b",
  "input": "What is the current NVIDIA stock price? Use the tool.",
  "max_output_tokens": 2048,
  "tools": [{"type": "function", "name": "get_stock_price",
             "description": "Get the latest stock price for a ticker symbol",
             "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}]
}' | jq '.output[] | select(.type=="function_call") | {name, arguments}'
```

Expected: `OK`, then `APPLE`, then a `get_stock_price` call with `{"ticker":"NVDA", ...}`.

Sending the second request straight to Dynamo (port 8000) instead of the gateway fails with `501`; that difference is
the value the gateway adds. The third request exercises a client-executed function tool: Dynamo returns the function
call and the application runs it.

## Recorded cassettes and CI

The interactions above are recorded in `crates/agentic-server-core/tests/cassettes/dynamo/` and replayed by
`tests/dynamo_cassette_test.rs` on every `cargo test`, so CI covers the Dynamo upstream without a GPU. The
`dynamo-upstream` CI job also runs `scripts/validate-cassettes.py`, a structural check over every recorded cassette. To re-record
against a live Dynamo (for example after a Dynamo release changes the response shape):

```bash
cd crates/agentic-server-core
DYNAMO_URL=http://127.0.0.1:8000 MODEL=openai/gpt-oss-20b \
  bash tests/cassettes/record_dynamo_cassettes.sh
```

The script records the second stateful turn from the hydrated item history the gateway would send (built from turn
1's recorded assistant message), because the recorder's own `previous_response_id` chaining cannot be used against a
stateless upstream.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `unrecognized arguments: --tool-call-parser` | Use `--dyn-tool-call-parser` on the worker. |
| Answer text begins with `analysis…assistantfinal…` | Add `--dyn-reasoning-parser gpt_oss` (or the parser for your model). |
| `Free memory on device … is less than desired GPU memory utilization` | Lower `--gpu-memory-utilization`; it is a fraction of total memory. |
| `CUDA error: out of memory` right after restarting a worker | A previous `dynamo.vllm` process is still alive and holding memory; `pkill -f "python -m dynamo.vllm"` before relaunching. Closing its terminal or tmux window does not kill it. |
| `501 Validation: previous_response_id is not supported.` | You are calling Dynamo directly. Send the request to the gateway. |
| Gateway logs `LLM ready` but requests fail with no model / `model not found` | `/health` is green before the worker registers. Check `/v1/models` or `/v1/models/{model}/ready` and the worker log. |
| Installed version is `1.5.0.dev…` | `--prerelease=allow` with an unpinned `ai-dynamo` picked a dev build. Reinstall with `"ai-dynamo[vllm]==1.4.1"`. |

## Messages recording preparation (#213)

The Messages recording workflow is prepared for a single Linux GPU host. **Dynamo Messages recordings have not yet
been captured for this change.** The local preparation tests replay the existing vLLM Messages recordings; they do
not establish Dynamo compatibility. The explicit `dynamo_messages_recorded_acceptance` test requires the real Dynamo
files and fails if they are absent.

The frontend must enable the experimental Anthropic endpoint with `--enable-anthropic-api`. The gateway sends
`/v1/messages` to that endpoint; it does not convert Messages requests into Chat Completions itself. Check this flag
with the pinned frontend's `--help` before recording. Keep the worker's `--dyn-reasoning-parser gpt_oss` and
`--dyn-tool-call-parser harmony` settings.

### Prepare and test on a MacBook

```bash
uv venv --python 3.12 .venv-dynamo-recorder
uv pip install --python .venv-dynamo-recorder/bin/python -r scripts/dynamo/recorder-requirements.txt
.venv-dynamo-recorder/bin/python -m unittest discover -s scripts/dynamo -p 'test_*.py' -v
cargo test --locked -p agentic-server-core --test dynamo_messages_test
```

The Python test runs the actual recorder against a local replay server, verifies the two-round history, and checks
that failure in the second recording mode leaves installed recordings unchanged. Its temporary captures are deleted
when the test finishes. The Rust tests exercise the production gateway tool loop with a fixed test executor whose
output matches `messages/tool_outputs.json`. No external search service, subscription, or GPU is used.

The existing streaming vLLM fixture predates recorder support for `signature_delta`: its captured response contains a
signature that its next request omitted. The preparation test derives that one legacy expectation from the captured
wire event. Dynamo acceptance has no such exception. New recordings preserve signatures and reject malformed tool
argument JSON, missing stream completion, and upstream error events instead of silently repairing them.

### Prepare one RunPod Pod

Use an on-demand Linux x86_64 Pod with a single L40S 48 GB, at least 8 vCPU and 64 GB host RAM, and approximately
150 GB workspace disk. This configuration has enough VRAM for `openai/gpt-oss-20b`; the host RAM and disk figures
leave room for compilation, model downloads, and retained logs. The
pinned Dynamo 1.4.1/vLLM 0.26.0 CUDA 13 combination requires an NVIDIA 580-series or newer host driver; a container
cannot replace an incompatible host driver. See the [Dynamo compatibility matrix](https://docs.nvidia.com/dynamo/dev/reference/compatibility).

All commands below run **inside the Pod**. Transfer the development checkout, including uncommitted changes, into
`/workspace/agentic-api` first. Keep model caches on persistent workspace storage. Do not copy a macOS `target/`
directory: Linux needs its own Rust build. Install `uv` and `rustup` from their official installers if absent; on an
Ubuntu template install the build prerequisites with:

```bash
apt-get update
apt-get install -y build-essential pkg-config libssl-dev ca-certificates git curl
cd /workspace/agentic-api
export HF_HOME=/workspace/huggingface
bash scripts/dynamo/prepare-pod.sh
```

The preparation script checks the driver and installs separate Dynamo and recorder environments. It uses the checked-in
Rust toolchain and locked Cargo dependencies, builds the gateway, and runs the local preparation tests. It does not
install or upgrade the host driver. Python recorder dependencies are pinned in `scripts/dynamo/recorder-requirements.txt`;
Dynamo's installed transitive versions are exported with the session artifacts.

Run the recording session:

```bash
.venv-dynamo-recorder/bin/python scripts/dynamo/run_pod.py --output /workspace/dynamo-session-01
```

The runner requires an unused output directory and unused ports 8000, 9000, and 7070. It starts the frontend and one
worker using file discovery, waits up to 15 minutes for model readiness, records both Messages modes, starts the
gateway for a client-executed function-tool smoke test, and explicitly runs the GPU-recording replay test. It stops
its own child process groups on success, failure, or interruption. No nested Docker, Kubernetes, etcd, or NATS is needed.
Use a dedicated Pod without other Dynamo processes: file discovery is host-local shared state.

Each mode records a `web_search` call followed by a fixed tool output and a final answer. The fixed output is a test
fixture, not a claim about today's Rust release. Both recordings must pass scenario and structural checks before
replacing the destination files. A failed run reports its staging directory for diagnosis. No YAML should be written
or corrected by hand. The model may fail to complete the scenario within the two-round/token budget; retain that
failure and investigate rather than loosening the assertions.

For a manually managed frontend, recording alone is:

```bash
PYTHON="$PWD/.venv-dynamo-recorder/bin/python" \
DYNAMO_URL=http://127.0.0.1:8000 MODEL=openai/gpt-oss-20b \
bash crates/agentic-server-core/tests/cassettes/record_dynamo_messages_cassettes.sh

cargo test --locked -p agentic-server-core --test dynamo_messages_test \
  dynamo_messages_recorded_acceptance -- --ignored --exact
```

The session directory retains selected environment metadata (source commit and dirty status, GPU/driver, model cache
snapshot, installed package versions, exact launch arguments), service logs, gateway smoke responses, and replay
output. Only a fully successful run writes `SUCCESS` and copies the accepted cassettes there. Retrieve the checkout
and session artifacts before terminating the Pod; storage may continue to be billed while a Pod is stopped.
No environment-variable dump or credentials are included by the exporter; inspect captured headers before sharing.

### Finish the PR after GPU validation

1. Inspect the real streaming and non-streaming captures and reproduce any provider differences with a minimal request.
2. Run scenario validation, the explicit Dynamo acceptance test above, the existing Dynamo/ Messages tests, full Rust
   tests, Clippy, formatting, and pre-commit checks.
3. Remove the acceptance test's temporary `ignore` attribute once real recordings are checked in, so the existing
   `dynamo-upstream` CI job runs it on every change. Remove the preparation-only status language from this guide and
   replace it with the exact verified GPU/software setup and observed compatibility findings.
4. Keep any unrelated Dynamo or gateway behavior change in a separately explained fix. A successful mock replay alone
   is not evidence of live Dynamo compatibility and is not sufficient to close #213.
