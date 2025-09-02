# Streaming Chat Completion

This guide covers how to use streaming chat completions in the Go LLM library. Streaming allows you to receive responses as they are generated, providing better user experience and real-time interactivity.

## Overview

Streaming chat completion enables real-time response generation where tokens are sent as they become available, rather than waiting for the complete response. This is particularly useful for:

- Interactive chat applications
- Long-form content generation
- Real-time user interfaces
- Applications requiring immediate feedback

## Basic Usage

### Simple Streaming Example

```go
package main
import (
    "context"
    "fmt"
    "log"
    "strings"

    "github.com/inercia/go-llm/pkg/llm"
    "github.com/inercia/go-llm/pkg/factory"
)

func main() {
    // Create factory and client
    factory := factory.New()
    client, err := factory.CreateClient(llm.ClientConfig{
        Provider: "openai",
        Model:    "gpt-3.5-turbo",
        APIKey:   "your-api-key",
    })
    if err != nil {
        log.Fatal("Failed to create client:", err)
    }
    defer client.Close()

    // Check streaming support
    modelInfo := client.GetModelInfo()
    if !modelInfo.SupportsStreaming {
        log.Fatal("Model does not support streaming")
    }

    // Create streaming request
    req := llm.ChatRequest{
        Model: modelInfo.Name,
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: []llm.MessageContent{
                llm.NewTextContent("Tell me a short story about AI."),
            }},
        },
        Stream: true, // Enable streaming
    }

    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    // Start streaming
    stream, err := client.StreamChatCompletion(ctx, req)
    if err != nil {
        log.Fatal("Failed to create stream:", err)
    }

    // Process streaming events
    var fullResponse strings.Builder
    fmt.Print("AI Story: ")

    for event := range stream {
        switch {
        case event.IsDelta():
            // Process incremental content
            if len(event.Choice.Delta.Content) > 0 {
                if textContent, ok := event.Choice.Delta.Content[0].(*llm.TextContent); ok {
                    chunk := textContent.GetText()
                    fullResponse.WriteString(chunk)
                    fmt.Print(chunk) // Display in real-time
                }
            }
        case event.IsDone():
            fmt.Printf("\n\nStream completed: %s\n", event.Choice.FinishReason)
            fmt.Printf("Full story: %s\n", fullResponse.String())
        case event.IsError():
            log.Printf("Stream error: %s (code: %s)", event.Error.Message, event.Error.Code)
        }
    }
}
```

## Stream Event Types

The streaming API provides different event types to handle various scenarios:

### Delta Events

- **Type**: `event.IsDelta()` returns `true`
- **Purpose**: Contains incremental content chunks
- **Usage**: Build up the complete response by concatenating delta content

### Done Events

- **Type**: `event.IsDone()` returns `true`
- **Purpose**: Indicates stream completion
- **Content**: Contains finish reason and final state

### Error Events

- **Type**: `event.IsError()` returns `true`
- **Purpose**: Reports streaming errors
- **Content**: Contains error details with message and code

## Advanced Streaming Patterns

### Streaming with Context and Timeout

```go
func streamWithTimeout(client llm.Client) error {
    ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
    defer cancel()

    req := llm.ChatRequest{
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: []llm.MessageContent{
                llm.NewTextContent("Explain quantum computing in detail."),
            }},
        },
        Stream: true,
    }

    stream, err := client.StreamChatCompletion(ctx, req)
    if err != nil {
        return fmt.Errorf("stream creation failed: %w", err)
    }

    for event := range stream {
        select {
        case <-ctx.Done():
            return fmt.Errorf("stream timeout: %w", ctx.Err())
        default:
            // Process event normally
            if event.IsError() {
                return fmt.Errorf("stream error: %s", event.Error.Message)
            }
            // Handle other events...
        }
    }

    return nil
}
```

### Streaming with Progress Tracking

```go
func streamWithProgress(client llm.Client, progressCallback func(int)) error {
    req := llm.ChatRequest{
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: []llm.MessageContent{
                llm.NewTextContent("Write a comprehensive guide on Go concurrency."),
            }},
        },
        Stream: true,
    }

    stream, err := client.StreamChatCompletion(context.Background(), req)
    if err != nil {
        return err
    }

    tokenCount := 0
    for event := range stream {
        if event.IsDelta() && len(event.Choice.Delta.Content) > 0 {
            if textContent, ok := event.Choice.Delta.Content[0].(*llm.TextContent); ok {
                // Rough token estimation (words * 1.3)
                words := len(strings.Fields(textContent.GetText()))
                tokenCount += int(float64(words) * 1.3)
                progressCallback(tokenCount)
            }
        }
    }

    return nil
}
```

### Concurrent Streaming

```go
func handleMultipleStreams(client llm.Client, queries []string) {
    var wg sync.WaitGroup
    results := make(chan string, len(queries))

    for i, query := range queries {
        wg.Add(1)
        go func(id int, q string) {
            defer wg.Done()

            req := llm.ChatRequest{
                Messages: []llm.Message{
                    {Role: llm.RoleUser, Content: []llm.MessageContent{
                        llm.NewTextContent(q),
                    }},
                },
                Stream: true,
            }

            stream, err := client.StreamChatCompletion(context.Background(), req)
            if err != nil {
                log.Printf("Stream %d failed: %v", id, err)
                return
            }

            var response strings.Builder
            for event := range stream {
                if event.IsDelta() && len(event.Choice.Delta.Content) > 0 {
                    if textContent, ok := event.Choice.Delta.Content[0].(*llm.TextContent); ok {
                        response.WriteString(textContent.GetText())
                    }
                }
            }

            results <- fmt.Sprintf("Stream %d: %s", id, response.String())
        }(i, query)
    }

    go func() {
        wg.Wait()
        close(results)
    }()

    for result := range results {
        fmt.Println(result)
    }
}
```

## Streaming with Tools

Streaming can be combined with tool calling functionality:

```go
func streamWithTools(client llm.Client) error {
    calculatorTool := llm.Tool{
        Type: "function",
        Function: llm.ToolFunction{
            Name:        "calculate",
            Description: "Perform basic arithmetic operations",
            Parameters: map[string]interface{}{
                "type": "object",
                "properties": map[string]interface{}{
                    "expression": map[string]interface{}{
                        "type":        "string",
                        "description": "Mathematical expression to evaluate",
                    },
                },
                "required": []string{"expression"},
            },
        },
    }

    req := llm.ChatRequest{
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: []llm.MessageContent{
                llm.NewTextContent("Calculate 123 * 456 and explain the result."),
            }},
        },
        Tools:  []llm.Tool{calculatorTool},
        Stream: true,
    }

    stream, err := client.StreamChatCompletion(context.Background(), req)
    if err != nil {
        return err
    }

    for event := range stream {
        if event.IsDelta() && event.Choice.Delta != nil {
            // Check for tool calls in streaming
            if event.Choice.Delta.ToolCalls != nil {
                for _, toolCall := range event.Choice.Delta.ToolCalls {
                    if toolCall.Function != nil {
                        fmt.Printf("Tool call: %s\n", toolCall.Function.Name)
                    }
                }
            }

            // Process text content
            for _, content := range event.Choice.Delta.Content {
                if textContent, ok := content.(*llm.TextContent); ok {
                    fmt.Print(textContent.GetText())
                }
            }
        }
    }

    return nil
}
```

## Error Handling and Recovery

### Robust Stream Processing

```go
func robustStreamHandling(client llm.Client, req llm.ChatRequest) (string, error) {
    const maxRetries = 3
    var lastError error

    for retry := 0; retry < maxRetries; retry++ {
        ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

        stream, err := client.StreamChatCompletion(ctx, req)
        if err != nil {
            cancel()
            lastError = err
            time.Sleep(time.Duration(retry+1) * time.Second) // Exponential backoff
            continue
        }

        var response strings.Builder
        var streamError error

        for event := range stream {
            switch {
            case event.IsError():
                streamError = fmt.Errorf("stream error: %s", event.Error.Message)
                break
            case event.IsDelta():
                if len(event.Choice.Delta.Content) > 0 {
                    if textContent, ok := event.Choice.Delta.Content[0].(*llm.TextContent); ok {
                        response.WriteString(textContent.GetText())
                    }
                }
            case event.IsDone():
                cancel()
                return response.String(), nil
            }
        }

        cancel()

        if streamError != nil {
            lastError = streamError
            time.Sleep(time.Duration(retry+1) * time.Second)
            continue
        }

        return response.String(), nil
    }

    return "", fmt.Errorf("streaming failed after %d retries: %w", maxRetries, lastError)
}
```

## Best Practices

### 1. Model Compatibility

Always check if the model supports streaming before attempting to use it:

```go
modelInfo := client.GetModelInfo()
if !modelInfo.SupportsStreaming {
    // Fall back to non-streaming or return error
    return errors.New("streaming not supported by this model")
}
```

### 2. Proper Context Management

Use contexts with timeouts to prevent indefinite blocking:

```go
ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
defer cancel()

stream, err := client.StreamChatCompletion(ctx, req)
```

### 3. Event Limit Protection

Implement safety limits to prevent infinite loops:

```go
eventCount := 0
maxEvents := 1000

for event := range stream {
    eventCount++
    if eventCount > maxEvents {
        log.Println("Event limit reached")
        break
    }
    // Process event...
}
```

### 4. Memory Management

For long streams, consider implementing buffer management:

```go
const maxBufferSize = 10 * 1024 * 1024 // 10MB limit

var response strings.Builder
for event := range stream {
    if event.IsDelta() && len(event.Choice.Delta.Content) > 0 {
        if textContent, ok := event.Choice.Delta.Content[0].(*llm.TextContent); ok {
            chunk := textContent.GetText()

            // Check buffer size
            if response.Len()+len(chunk) > maxBufferSize {
                return "", errors.New("response too large")
            }

            response.WriteString(chunk)
        }
    }
}
```

### 5. Graceful Shutdown

Implement proper cleanup for streaming operations:

```go
func gracefulStreamShutdown(stream <-chan llm.StreamEvent, done chan<- bool) {
    defer close(done)

    for event := range stream {
        select {
        case <-time.After(100 * time.Millisecond):
            // Process event with timeout
            processEvent(event)
        default:
            // Skip if processing takes too long
            continue
        }
    }
}
```

## Performance Considerations

### 1. Network Latency

- Streaming reduces perceived latency by showing results immediately
- Consider implementing client-side buffering for smoother display

### 2. Processing Overhead

- Each stream event has overhead; balance real-time updates with performance
- Consider batching small chunks for display updates

### 3. Concurrent Streams

- Limit concurrent streaming requests to avoid overwhelming the API
- Implement connection pooling and rate limiting

### 4. Resource Management

- Always close clients properly to free resources
- Monitor memory usage during long streaming sessions

## Common Issues and Solutions

### Issue: Stream Hangs

**Cause**: Network issues, API problems, or missing context timeouts

**Solution**:

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

// Monitor context in stream processing
select {
case event, ok := <-stream:
    if !ok {
        return // Stream closed
    }
    // Process event
case <-ctx.Done():
    return fmt.Errorf("stream timeout: %w", ctx.Err())
}
```

### Issue: Missing Content

**Cause**: Incorrect event type checking or content extraction

**Solution**:

```go
if event.IsDelta() && event.Choice != nil && event.Choice.Delta != nil {
    for _, content := range event.Choice.Delta.Content {
        if textContent, ok := content.(*llm.TextContent); ok && textContent.GetText() != "" {
            // Process non-empty text content
            fmt.Print(textContent.GetText())
        }
    }
}
```

### Issue: Memory Leaks

**Cause**: Not properly closing streams or accumulating too much data

**Solution**:

- Use `defer client.Close()`
- Implement size limits on response builders
- Clear temporary buffers regularly

## See Also

- [Tools Documentation](tools.md) - Combining streaming with tool functionality
- [Multimodal Documentation](multimodal.md) - Streaming with images and files
- [Architecture Overview](architecture.md) - Understanding the client architecture
- [Examples Directory](../examples/) - Complete runnable examples
