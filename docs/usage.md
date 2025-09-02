# Usage Guide

This guide covers the basic usage patterns for the LLM client library. It assumes you have the package imported as `github.com/inercia/go-llm/pkg/llm`.

## Installation

Add the package to your Go module:

```bash
go get github.com/inercia/go-llm
```

## Basic Chat Completion (Non-Streaming)

Send a chat request and receive a full response.

```go
ctx := context.Background()

resp, err := client.ChatCompletion(ctx, llm.ChatRequest{
    Model: "gemini-1.5-flash",
    Messages: []llm.Message{
        {Role: llm.RoleUser, Content: "Explain quantum computing in simple terms."},
    },
})
if err != nil {
    log.Fatal(err)
}

if len(resp.Choices) > 0 {
    fmt.Println(resp.Choices[0].Message.GetText())
}
```

## Error Handling

All errors are standardized as `llm.Error` with fields like Code, Message, Type, and StatusCode.

```go
resp, err := client.ChatCompletion(ctx, req)
if err != nil {
    if llmErr, ok := err.(*llm.Error); ok {
        fmt.Printf("Error: %s (Code: %s, Status: %d)\n", llmErr.Message, llmErr.Code, llmErr.StatusCode)
    } else {
        log.Fatal(err)
    }
}
```

**Gemini Error Handling Example**:

```go
// Handles both object and array error formats
if resp.StatusCode != http.StatusOK {
    return nil, c.convertGeminiError(body, resp.StatusCode)
}

func (c *GeminiClient) convertGeminiError(body []byte, statusCode int) *Error {
    // Try object format: {"error": {...}}
    var geminiErr GeminiError
    if err := json.Unmarshal(body, &geminiErr); err == nil {
        return standardizeError(geminiErr)
    }

    // Try array format: [{"error": {...}}]
    var geminiErrArray []GeminiError
    if err := json.Unmarshal(body, &geminiErrArray); err == nil {
        return standardizeError(geminiErrArray[0])
    }

    return fallbackError(body, statusCode)
}
```

## Common Patterns

- **Multi-turn Conversations**: Append previous messages to the Messages array with appropriate roles (system, user, assistant).
- **Tool Calls**: If supported, include tools in the request and handle tool calls in responses. See [Tools & Function Calling](tools.md) for detailed examples.
- **Testing**: Use the "mock" provider for unit tests without real API calls.

For provider-specific details, see the [Providers documentation](README.md#providers).

## Next Steps

Once you're comfortable with basic usage, explore these advanced capabilities:

- **[Streaming](streaming.md)**: Real-time response generation for better user experience
- **[Tools & Function Calling](tools.md)**: Enable LLMs to call external functions and APIs
- **[Multimodal Messages](multimodal.md)**: Work with images, files, and mixed content types
- **[Advanced Patterns](advanced.md)**: Structured outputs, retries, and other advanced features
