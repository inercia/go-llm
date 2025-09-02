# Ollama Provider

The Ollama provider connects directly to a local Ollama server via HTTP, enabling offline LLM usage without cloud dependencies. It implements the core `Client` interface using Ollama's API.

## Features

- **Local Execution**: Runs entirely on your machine; no internet or API keys required.
- **Chat Completions**: Supports multi-turn conversations with Ollama-compatible models.
- **Streaming**: NDJSON-based streaming for real-time responses.
- **Model Management**: Can list and use models pulled via Ollama CLI; library checks server reachability.
- **Error Standardization**: Maps Ollama error responses (e.g., model not found) to `llm.Error`.
- **Customizable**: Configurable base URL (default: http://localhost:11434), model names from Ollama hub.
- **Lightweight**: No external dependencies beyond standard HTTP client.

## Setup

1. Install Ollama from [ollama.com](https://ollama.com/download) and start the server:

```bash
ollama serve
```

2. Pull a model (e.g., Llama 3):

```bash
ollama pull llama3
```

3. Create the client (no API key needed):

```go
client, err := factory.CreateClient(llm.ClientConfig{
    Provider: "ollama",
    Model:    "llama3",  // Or other pulled models like "phi3", "mistral"
    BaseURL:  "http://localhost:11434",  // Optional, defaults to this
})
if err != nil {
    log.Fatal(err)  // e.g., if server unreachable
}
```

Supported models: Any from [Ollama library](https://ollama.com/library), quantized for local hardware (e.g., 7B, 13B params).

## Known Issues and Workarounds

- **Server Unreachable**: If Ollama not running, client creation fails with connection error. Start `ollama serve` first.
- **Model Not Found**: 404 if model not pulled; run `ollama pull <model>` or list with `ollama list`.
- **Streaming Format**: Uses NDJSON; rare parsing issues if server misconfigured—ensure Ollama is up-to-date.
- **High Latency**: Initial model loading can take seconds; subsequent requests faster. Use smaller models for speed.
- **GPU Acceleration**: If available (NVIDIA/Apple Silicon), Ollama auto-detects; otherwise falls back to CPU (slower).
- **Memory Leaks**: Long-running sessions may consume memory; restart Ollama periodically for heavy use.
- **Version Compatibility**: Library tested with Ollama v0.1+; update Ollama for bug fixes.

For testing, run integration tests without env vars (uses local check). Mock client simulates Ollama responses.

See the [main usage guide](../usage.md) for general examples.
