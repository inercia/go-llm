# Usage Guide

This guide covers the basic usage patterns for the LLM client library. It assumes you have the package imported as `github.com/inercia/go-llm/pkg/llm`.

## Installation

Add the package to your Go module:

```bash
go get github.com/inercia/go-llm
```

## Initializing a Client

Use the factory to create clients for different providers. Set the appropriate API key via environment variable or directly in config.

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    "github.com/inercia/go-llm/pkg/llm"
)

func main() {
    factory := llm.NewFactory()
    client, err := factory.CreateClient(llm.ClientConfig{
        Provider: "gemini",  // or "openai", "ollama"
        APIKey:   "your-api-key",  // For Ollama, this can be empty if running locally
        Model:    "gemini-1.5-flash",  // Provider-specific model name
        Timeout:  30 * time.Second,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()
}
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

## Streaming Chat Completion

For real-time responses, use streaming if supported by the model/provider.

First, check support:

```go
modelInfo := client.GetModelInfo()
if modelInfo.SupportsStreaming {
    // Proceed with streaming
}
```

Streaming example:

```go
stream, err := client.StreamChatCompletion(ctx, llm.ChatRequest{
    Model: "gemini-1.5-flash",
    Messages: []llm.Message{
        {Role: llm.RoleUser, Content: "Tell me a story about AI."},
    },
    Stream: true,
})
if err != nil {
    log.Fatal(err)
}

var fullResponse string
for event := range stream {
    if event.IsDelta() && len(event.Choice.Delta.Content) > 0 {
        text := event.Choice.Delta.Content[0].GetText()
        fullResponse += text
        fmt.Print(text)
    } else if event.IsDone() {
        fmt.Println("\nStream complete. Finish reason:", event.Choice.FinishReason)
    } else if event.IsError() {
        log.Fatal("Stream error:", event.Error.Message)
    }
}
fmt.Println("\nFull response:", fullResponse)
```

**StreamEvent Types:**

- **Delta Events**: Incremental content updates (`event.IsDelta()`)
  - `event.Choice.Delta.Content`: New text chunks
  - `event.Choice.Delta.ToolCalls`: Incremental tool call details

- **Done Events**: Stream completion (`event.IsDone()`)
  - `event.Choice.FinishReason`: "stop", "length", "tool_calls", etc.

- **Error Events**: Stream errors (`event.IsError()`)
  - `event.Error`: Standardized error details

**Provider Support:**

- **OpenAI**: Full streaming support with tool calls
- **Gemini**: Streaming text and function calls
- **OpenRouter**: Multi-provider streaming with tool calls
- **Ollama**: NDJSON streaming for local models
- **Mock**: Simulated streaming for testing

The `SupportsStreaming` field in `ModelInfo` indicates if streaming is available for the model. 2. **Embeddings**: Support for text embeddings across providers 3. **Fine-tuning**: Abstraction for model fine-tuning APIs 4. **Observability**: Built-in metrics and tracing 5. **Caching**: Response caching layer 6. **Rate Limiting**: Built-in rate limiting and retry logic

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

## Model Information

Retrieve details about the current model:

```go
modelInfo := client.GetModelInfo()
fmt.Printf("Model: %s, Supports Streaming: %t\n", modelInfo.Name, modelInfo.SupportsStreaming)
```

## Common Patterns

- **Multi-turn Conversations**: Append previous messages to the Messages array with appropriate roles (system, user, assistant).
- **Tool Calls**: If supported, include tools in the request and handle tool calls in responses.
- **Testing**: Use the "mock" provider for unit tests without real API calls.

For provider-specific details, see the [Providers documentation](README.md#providers).

See [examples/streaming_example.go](../examples/streaming_example.go) for a complete runnable example.
