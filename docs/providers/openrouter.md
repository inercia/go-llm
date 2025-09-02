# OpenRouter Provider

The OpenRouter provider integrates with OpenRouter's multi-provider API using the `go-openrouter` library. It provides access to multiple LLM providers through a single unified interface, making it easy to switch between different models and providers.

## Features

- **Multi-Provider Access**: Access to OpenAI, Anthropic, Google, Meta, and many other providers through a single API.
- **Chat Completions**: Full support for multi-turn conversations with system, user, and assistant messages.
- **Streaming**: Real-time token-by-token responses via Server-Sent Events (SSE).
- **Tool/Function Calling**: Native support for function calling with compatible models.
- **Multi-modal Support**: Vision capabilities for models that support image inputs.
- **File Support**: Handle file inputs for compatible models.
- **Error Standardization**: Maps OpenRouter's error responses to the library's `llm.Error` structure.
- **Model Information**: Retrieves details about model capabilities and limits.
- **Custom Configuration**: Support for site URL and app name headers for better analytics.

## Setup

1. Obtain an OpenRouter API key from [openrouter.ai](https://openrouter.ai/keys).
2. Set the `OPENROUTER_API_KEY` environment variable or pass it in `ClientConfig.APIKey`.
3. Use the factory to create the client:

```go
client, err := factory.CreateClient(llm.ClientConfig{
    Provider: "openrouter",
    APIKey:   "sk-or-your-openrouter-api-key",
    Model:    "openai/gpt-4o-mini",  // or any supported model
})
```

### Advanced Configuration

OpenRouter supports additional configuration options through the `Extra` field:

```go
client, err := factory.CreateClient(llm.ClientConfig{
    Provider: "openrouter",
    APIKey:   "sk-or-your-openrouter-api-key",
    Model:    "anthropic/claude-3-sonnet",
    BaseURL:  "https://openrouter.ai/api/v1",  // Optional custom base URL
    Extra: map[string]string{
        "site_url": "https://myapp.com",     // For analytics and rate limiting
        "app_name": "MyApplication",         // App identifier
    },
})
```

## Supported Models

OpenRouter provides access to models from multiple providers. Popular models include:

- **OpenAI**: `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-3.5-turbo`
- **Anthropic**: `anthropic/claude-3-opus`, `anthropic/claude-3-sonnet`, `anthropic/claude-3-haiku`
- **Google**: `google/gemini-pro`, `google/gemini-pro-vision`
- **Meta**: `meta-llama/llama-3-70b-instruct`, `meta-llama/llama-3-8b-instruct`
- **Mistral**: `mistralai/mistral-7b-instruct`, `mistralai/mixtral-8x7b-instruct`

For a complete list of available models, visit [openrouter.ai/models](https://openrouter.ai/models).

## Known Issues and Workarounds

- **Authentication Errors**: 401 errors if API key is invalid or has insufficient credits. Check your key and account balance at openrouter.ai.
- **Rate Limiting**: Returns 429 for rate limits; implement exponential backoff for retries. Rate limits vary by model and account tier.
- **Model Availability**: Some models may be temporarily unavailable due to provider issues. OpenRouter returns 503 in these cases.
- **Credit Limits**: Requests fail if account credits are exhausted. Monitor usage at openrouter.ai/activity.
- **Model-Specific Limitations**: Each underlying provider has different capabilities:
  - Not all models support function calling or vision
  - Context windows vary significantly between models
  - Some models have content filtering that may reject certain inputs
- **File Support Limitations**: OpenRouter only supports binary file data, not file URLs. Files must be base64-encoded.
- **Streaming Interruptions**: Network issues can close streams; handle `event.IsError()` in streaming loops.
- **Provider-Specific Errors**: Different underlying providers may return different error formats, but the library standardizes them.
- **Model Routing**: OpenRouter may route requests to different model versions; specify exact model versions if consistency is critical.

## Error Handling

The OpenRouter provider maps various error conditions to standardized error types:

- **Authentication Errors**: Invalid API keys, insufficient permissions
- **Rate Limit Errors**: Too many requests, quota exceeded
- **Model Errors**: Model not found, model overloaded, model not supported
- **Validation Errors**: Invalid request format, content filtered, token limits exceeded
- **Network Errors**: Connection issues, timeouts, DNS problems

Example error handling:

```go
resp, err := client.ChatCompletion(ctx, req)
if err != nil {
    if llmErr, ok := err.(*llm.Error); ok {
        switch llmErr.Type {
        case "rate_limit_error":
            // Implement backoff and retry
            time.Sleep(time.Second * 5)
        case "authentication_error":
            // Check API key and account status
            log.Fatal("Invalid API key or insufficient credits")
        case "model_error":
            // Try a different model
            log.Printf("Model issue: %s", llmErr.Message)
        }
    }
}
```

For testing, use integration tests with `OPENROUTER_API_KEY` set. Mock client can simulate OpenRouter responses.

See the [main usage guide](../usage.md) for general examples.
