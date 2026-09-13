#!/usr/bin/env bash
# Run inside an Ubuntu GPU Pod. Installs user-space dependencies, never the host driver.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Requires Linux x86_64' >&2; exit 1; }
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
[[ "$major" -ge 580 ]] || { echo 'Dynamo 1.4.1 requires driver 580 or newer; select another host.' >&2; exit 1; }
for cmd in uv cargo cc pkg-config; do
    command -v "$cmd" >/dev/null || { echo "Missing $cmd; see docs/guides/dynamo-upstream.md" >&2; exit 1; }
done
uv venv --allow-existing --python 3.12 .venv-dynamo
uv pip install --python .venv-dynamo/bin/python 'ai-dynamo[vllm]==1.4.1'
uv venv --allow-existing --python 3.12 .venv-dynamo-recorder
uv pip install --python .venv-dynamo-recorder/bin/python -r scripts/dynamo/recorder-requirements.txt
cargo build --locked -p agentic-server --bins
cargo test --locked -p agentic-server-core --test dynamo_messages_test
.venv-dynamo-recorder/bin/python -m unittest discover -s scripts/dynamo -p 'test_*.py'
echo 'Prepared. Next: .venv-dynamo-recorder/bin/python scripts/dynamo/run_pod.py'
