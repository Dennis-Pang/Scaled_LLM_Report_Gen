# SGLang Client

OpenAI-compatible client for SGLang VLM (e.g., Qwen3-VL-32B-Instruct).

## Features

- OpenAI-compatible API for text and multimodal chat
- Automatic server startup and model loading
- GPU selection and resource management
- Support for local image files and data URLs
- CLI utilities for server management

## Installation

```bash
pip install openai
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_BASE_URL` | `http://127.0.0.1:30000/v1` | SGLang server base URL |
| `SGLANG_MODEL` | `Qwen/Qwen3-VL-32B-Instruct` | Default model ID |
| `SGLANG_API_KEY` | `EMPTY` | API key (OpenAI-compatible) |
| `SGLANG_HOST` | `127.0.0.1` | Host to bind for server |
| `SGLANG_PORT` | `30000` | Port to bind for server |
| `SGLANG_START_TIMEOUT` | `300` | Seconds to wait for model load |
| `SGLANG_DISABLE_CUDNN_CHECK` | `True` | Disable CuDNN compatibility check |

## Usage

### Basic Text Chat

```python
from LLM.client import SGLangLLM

llm = SGLangLLM()
response = llm.chat("Hello, how are you?")
print(response)
```

### Multimodal Chat with Images

```python
from LLM.client import SGLangLLM

llm = SGLangLLM()
response = llm.chat_with_images(
    prompt="What's in this image?",
    images=["path/to/image.png"]
)
print(response)
```

### With System Prompt

```python
response = llm.chat(
    prompt="Summarize this text...",
    system_prompt="You are a helpful assistant.",
    temperature=0.3,
    max_tokens=512
)
```

### Custom Configuration

```python
llm = SGLangLLM(
    base_url="http://127.0.0.1:30001/v1",
    model="Qwen/Qwen3-VL-32B-Instruct",
    api_key="EMPTY",
    temperature=0.7,
    max_tokens=2048,
    gpu_ids=[0, 1],
    auto_start=False
)
```

## API Reference

### `SGLangLLM`

Main client class for interacting with SGLang VLM.

#### Methods

- `chat(prompt, system_prompt=None, temperature=None, max_tokens=None)`: Text-only chat
- `chat_with_images(prompt_or_content, images=None, system_prompt=None, temperature=None, max_tokens=None)`: Multimodal chat
- `chat_messages(messages, temperature=None, max_tokens=None)`: Chat with pre-built messages

### Utility Functions

- `list_models(base_url, api_key)`: List available model IDs
- `check_model_loaded(model_name, base_url, api_key)`: Check if model is loaded
- `test_connection(base_url, api_key)`: Test server connectivity
- `get_available_gpus()`: Get GPU information using nvidia-smi
- `select_gpus_for_model(num_gpus, available_gpus)`: Interactive GPU selection
- `start_server(model_name, gpu_ids, host, port, extra_args, disable_cudnn_check)`: Start SGLang server
- `ensure_server(model_name, base_url, api_key, auto_start, ...)`: Ensure server is running

## CLI Commands

### Client (server management)

```bash
# List available GPUs
python -m LLM.client gpus

# Check if model is loaded
python -m LLM.client check --model Qwen/Qwen3-VL-32B-Instruct

# Test server connection
python -m LLM.client test --base-url http://127.0.0.1:30000/v1

# Start SGLang server
python -m LLM.client start --model Qwen/Qwen3-VL-32B-Instruct --gpus 0,1
```

### Chat (quick testing)

```bash
# Interactive text chat (empty line to exit)
python -m LLM.chat

# One-shot text chat
python -m LLM.chat --text "Hello"

# Multimodal chat (repeat --image for multiple files)
python -m LLM.chat --text "Describe this image" --image /path/to/image.png

# Auto-start server if needed (will prompt for GPUs unless --gpu-ids provided)
python -m LLM.chat --auto-start --gpu-ids 0,1 --text "Hello"
```

## Supported Image Formats

- PNG (`.png`)
- JPEG/JPG (`.jpg`, `.jpeg`)
- GIF (`.gif`)
- WebP (`.webp`)

## Notes

- Images are converted to base64 data URLs for transmission
- The client automatically excludes `127.0.0.1` and `localhost` from proxy settings
- Default timeout for model loading is 300 seconds (5 minutes)
