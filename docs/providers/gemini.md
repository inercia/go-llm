# Gemini Provider

The Gemini provider integrates with Google's Gemini API using a native HTTP client. It provides a clean abstraction over the core `Client` interface.

## Features

- **Chat Completions**: Full support for multi-turn conversations with system, user, and assistant messages.
- **Streaming**: Real-time token-by-token responses via Server-Sent Events (SSE).
- **Error Standardization**: Handles Gemini's unique error formats (both single error object and error array) and maps them to the library's `llm.Error` structure.
- **Model Information**: Retrieves details like model name and streaming support.
- **Function/Tool Calling**: Supports Gemini's function calling for tool integrations.
- **Timeout and Retry**: Configurable timeouts; basic retry logic for transient errors.

## Setup

1. Obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Set the `GEMINI_API_KEY` environment variable or pass it in `ClientConfig.APIKey`.
3. Use the factory to create the client:

```go
client, err := factory.CreateClient(llm.ClientConfig{
    Provider: "gemini",
    APIKey:   "your-gemini-api-key",
    Model:    "gemini-1.5-flash",  // or "gemini-1.5-pro"
})
```

Supported models include `gemini-1.5-flash`, `gemini-1.5-pro`, and others available via the API.

## Known Issues and Workarounds

- **Error Format Variability**: Gemini may return errors as `{"error": {...}}` or `[{"error": {...}}]`. The library automatically detects and standardizes both.
- **Authentication Errors**: If API key is invalid, expect 401 Unauthorized. Ensure key has correct permissions.
- **Rate Limit Errors**: Returns 429; implement exponential backoff in your app for retries.
- **Streaming Interruptions**: Rare network issues can close streams; handle `event.IsError()` in loops.
- **Function Calling Differences**: Gemini's tool calls use "functionCalling" config; compatibility with OpenAI-style may require prompt adjustments.
- **Latency**: Initial requests may have higher latency (200-500ms) due to cold starts.

For testing, use integration tests with `GEMINI_API_KEY` set. Mock client can simulate Gemini responses.

See the [main usage guide](../usage.md) for general examples.
