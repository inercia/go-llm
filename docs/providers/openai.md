# OpenAI Provider

The OpenAI provider leverages the official `go-openai` library for seamless integration with OpenAI's API. It abstracts the core `Client` interface for compatibility across providers.

## Features

- **Chat Completions**: Support for GPT models with multi-turn conversations, system prompts, and JSON mode.
- **Streaming**: Full SSE streaming for real-time responses, including tool call streaming.
- **Tool/Function Calling**: Native support for parallel tool calls and structured outputs.
- **Model Information**: Fetches model details like context window and capabilities.
- **Error Standardization**: Maps OpenAI error codes (e.g., 429 rate limit) to `llm.Error`.
- **Advanced Options**: Temperature, top_p, max_tokens, and other sampling parameters configurable via `ChatRequest`.
- **Vision Support**: For models like gpt-4o, accepts image inputs in messages.

## Setup

1. Obtain an OpenAI API key from [platform.openai.com](https://platform.openai.com/api-keys).
2. Set the `OPENAI_API_KEY` environment variable or pass it in `ClientConfig.APIKey`.
3. Install the dependency (handled by Go modules):

```go
client, err := factory.CreateClient(llm.ClientConfig{
    Provider: "openai",
    APIKey:   "your-openai-api-key",
    Model:    "gpt-4o-mini",  // or "gpt-4o", "gpt-3.5-turbo"
})
```

Supported models: All chat-capable GPT models (e.g., `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`).

## Known Issues and Workarounds

- **Authentication Failures**: 401 errors if key is invalid/expired. Regenerate key if needed.
- **Rate Limiting**: 429 responses include retry-after header; implement client-side retries with backoff.
- **Token Overflow**: If input exceeds context, returns 400; trim messages or use summarization.
- **Streaming Tool Calls**: Delta events may arrive out-of-order for parallel tools; accumulate until done event.
- **JSON Mode Strictness**: When `response_format: {type: "json_object"}`, ensure prompt instructs JSON output.
- **Vision Input Limits**: Images must be base64-encoded or URL; size limits apply (20MB max).
- **Organization Headers**: If using organizations, set `OpenAIOOrganization` in config if needed.

For testing, use `OPENAI_API_KEY` with integration tests or mock client. Note: Real API calls incur costs.

See the [main usage guide](../usage.md) for general examples.
