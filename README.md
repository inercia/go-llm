# Flexible, multi-provider LLM client library for Go

## Documentation

For comprehensive guides, usage examples, and provider-specific details, see the [docs/README.md](docs/README.md).

This package provides a **clean and simplified** abstraction layer for Large Language Model (LLM)
clients, decoupling your code from specific LLM provider implementations.

## 🏗️ Architecture

### Core Interface

All LLM providers implement the following interface:

```go
type Client interface {
    // ChatCompletion performs a chat completion request
    ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error)

    // StreamChatCompletion performs a streaming chat completion request
    StreamChatCompletion(ctx context.Context, req ChatRequest) (<-chan StreamEvent, error)

    // GetModelInfo returns information about the model being used
    GetModelInfo() ModelInfo

    // Close cleans up any resources used by the client
    Close() error
}
```

### Provider Implementations

- **OpenAI Client** - Uses `go-openai` library internally
- **Gemini Client** - Native HTTP implementation with proper error handling
- **OpenRouter Client** - Multi-provider access via `go-openrouter` library
- **Mock Client** - For testing and development

### Simple Factory Pattern

```go
import (
    "github.com/inercia/go-llm/pkg/llm"
    "github.com/inercia/go-llm/pkg/factory"
)

// Create client using factory
factory := factory.New()
client, err := factory.CreateClient(llm.ClientConfig{
    Provider: "gemini",
    APIKey:   "your-api-key",
    Model:    "gemini-1.5-flash",
})
if err != nil {
    log.Fatal(err)
}
defer client.Close()

// Make request
resp, err := client.ChatCompletion(ctx, llm.ChatRequest{
    Model: "gemini-1.5-flash",
    Messages: []llm.Message{
        {Role: llm.RoleUser, Content: "Hello!"},
    },
})

if err != nil {
    log.Fatal(err)
}

fmt.Println(resp.Choices[0].Message.GetText())
```

### Streaming Support

The library supports streaming chat completions for all providers. The Client interface includes `StreamChatCompletion` method that returns a channel of `StreamEvent`.

**Usage:**

```go
// Check if streaming is supported
modelInfo := client.GetModelInfo()
if modelInfo.SupportsStreaming {
    stream, err := client.StreamChatCompletion(ctx, llm.ChatRequest{
        Model:   modelInfo.Name,
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: "Explain quantum computing in simple terms."},
        },
        Stream: true,
    })
    if err != nil {
        log.Fatal(err)
    }

    var fullResponse strings.Builder
    for event := range stream {
        switch {
        case event.IsDelta():
            if len(event.Choice.Delta.Content) > 0 {
                text := event.Choice.Delta.Content[0].GetText()
                fullResponse.WriteString(text)
                fmt.Print(text) // Real-time output
            }
            if len(event.Choice.Delta.ToolCalls) > 0 {
                // Handle streaming tool calls if needed
                for _, tc := range event.Choice.Delta.ToolCalls {
                    fmt.Printf("\nTool call: %s\n", tc.Function.Name)
                }
            }
        case event.IsDone():
            fmt.Println("\n\nStream complete. Finish reason:", event.Choice.FinishReason)
            fmt.Println("Full response:", fullResponse.String())
        case event.IsError():
            log.Printf("Stream error: %s", event.Error.Message)
        }
    }
} else {
    fmt.Println("Streaming not supported for this model")
    // Fall back to non-streaming
    resp, _ := client.ChatCompletion(ctx, req)
    fmt.Println(resp.Choices[0].Message.GetText())
}
```
