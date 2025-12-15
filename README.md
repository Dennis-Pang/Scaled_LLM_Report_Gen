## Scaled LLM Report Generation

This workspace mirrors the [`Dennis-Pang/Scaled_LLM_Report_Gen`](https://github.com/Dennis-Pang/Scaled_LLM_Report_Gen) project.
The goal is to run a local SGLang inference server for the Qwen3-32B model and
send structured extraction requests (see `GenAI/main.py`) through its
OpenAI-compatible API.

### Repository layout

```
GenAI/main.py        # sample client that hits the local SGLang endpoint
start_sglang.sh      # helper script to launch SGLang (Qwen3-32B, TP=2)
stop_sglang.sh       # helper script to stop the running server
sglang_qwen3_32b.log # default log file for the server (created on first run)
```

### Requirements

1. NVIDIA GPUs that can host the model shards (here we use two L40S GPUs).
2. Python environment `agentic` inside `/home/zixian.pang/miniconda3` with
   `sglang`, `torch`, `openai` etc. already installed.
3. The weights downloaded to `/home/zixian.pang/.model_cache/Qwen/Qwen3-32B`.

### Running the SGLang server

To launch the server with the default settings (tensor parallel size 2,
listening on port `30000`) run:

```bash
bash start_sglang.sh
```

`start_sglang.sh`:

- Activates the `agentic` conda env.
- Sets `NO_PROXY=127.0.0.1,localhost,0.0.0.0` so health checks bypass the
  corporate proxy.
- Executes:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
    --model-path /home/zixian.pang/.model_cache/Qwen/Qwen3-32B \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 2 \
    --trust-remote-code
  ```
- Redirects logs to `~/sglang_qwen3_32b.log` and stores the PID in
  `~/sglang_server.pid`.

Optional adjustments can be made by exporting environment variables before
invoking the script:

- `CUDA_VISIBLE_DEVICES` – choose which GPUs participate in tensor parallelism.
- `SGLANG_PORT` (if you extend the script) – listen on a different port.
- `SGLANG_MODEL_PATH` – point to another model directory.

To stop the server:

```bash
bash stop_sglang.sh
```

The script first reads the PID file and falls back to `pkill -f sglang.launch_server`
if the PID is missing/stale.

### Running the client

`GenAI/main.py` demonstrates how to call the local server via the OpenAI Chat
Completions API:

```bash
source ~/miniconda3/bin/activate agentic
NO_PROXY=127.0.0.1,localhost,0.0.0.0 \
  python 'GenAI/ main.py'
```

Key details:

- Uses `OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:30000/v1")`.
- Sends a medical note and expects JSON output with diagnosis, medications,
  allergies, and follow-up fields (`response_format={"type": "json_object"}`).
- Supports overriding defaults via environment variables:
  - `SGLANG_OPENAI_URL` – base URL of the OpenAI-compatible endpoint.
  - `SGLANG_MODEL_NAME` – value sent in the `model=` field.
  - `SGLANG_API_KEY` – forwarded as the API key.

The script prints the structured JSON response to stdout, which is the data
you can feed into downstream tooling or reporting pipelines.

### Example output

Below is a sample structured JSON response:

```json
{"diagnosis":"Type 2 diabetes with mild neuropathy","medications":["metformin 1g BID","gabapentin 100mg nightly"],"allergies":["penicillin"],"follow_up":"4 weeks"}
```
