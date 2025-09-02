# go-llm Architecture Documentation

## Overview

The `go-llm` library is a comprehensive, provider-agnostic abstraction layer for Large Language Model (LLM) clients in Go. It provides a unified interface for interacting with multiple LLM providers while supporting advanced features like multimodal content, streaming responses, tool calling, and message routing.

## Core Design Principles

1. **Provider Agnosticism**: Unified interface that abstracts away provider-specific implementations
2. **Multimodal Support**: Built-in support for text, images, and file content
3. **Streaming First**: Native support for streaming responses with proper event handling
4. **Type Safety**: Strongly typed interfaces with comprehensive validation
5. **Extensibility**: Plugin-based architecture for content handlers and message routers
6. **Thread Safety**: All components designed for safe concurrent usage

## System Architecture

### High-Level Component Overview

```mermaid
graph TB
    Client[Client Application] --> Factory[Factory]
    Factory --> |Creates| LLMClient[LLM Client Interface]

    LLMClient --> OpenAI[OpenAI Client]
    LLMClient --> Ollama[Ollama Client]
    LLMClient --> Gemini[Gemini Client]
    LLMClient --> OpenRouter[OpenRouter Client]
    LLMClient --> DeepSeek[DeepSeek Client]

    OpenAI --> |HTTP API| OpenAIAPI[OpenAI API]
    Ollama --> |HTTP API| OllamaAPI[Ollama Local API]
    Gemini --> |HTTP API| GeminiAPI[Google Gemini API]
    OpenRouter --> |HTTP API| OpenRouterAPI[OpenRouter API]
    DeepSeek --> |HTTP API| DeepSeekAPI[DeepSeek API]

    LLMClient --> MessageRouter[Message Router]
    MessageRouter --> ContentHandlers[Content Handlers]
    ContentHandlers --> TextHandler[Text Handler]
    ContentHandlers --> ImageHandler[Image Handler]
    ContentHandlers --> FileHandler[File Handler]

    style Client fill:#e1f5fe
    style Factory fill:#f3e5f5
    style LLMClient fill:#e8f5e8
    style MessageRouter fill:#fff3e0
    style ContentHandlers fill:#fce4ec
```

### Core Interfaces and Data Flow

```mermaid
graph LR
    subgraph "Request Flow"
        CR[ChatRequest] --> |Contains| Messages[Messages Array]
        Messages --> |Contains| MC[MessageContent]
        MC --> TC[TextContent]
        MC --> IC[ImageContent]
        MC --> FC[FileContent]
    end

    subgraph "Processing"
        CR --> Client[Client Interface]
        Client --> |Normal| ChatCompletion[ChatCompletion Method]
        Client --> |Stream| StreamChat[StreamChatCompletion Method]
    end

    subgraph "Response Flow"
        ChatCompletion --> ChatResponse[ChatResponse]
        StreamChat --> StreamEvents[Stream Events Channel]
        StreamEvents --> Delta[Delta Events]
        StreamEvents --> Done[Done Events]
        StreamEvents --> Error[Error Events]
    end

    style CR fill:#e3f2fd
    style Client fill:#e8f5e8
    style ChatResponse fill:#f1f8e9
    style StreamEvents fill:#fff8e1
```

## Component Details

### 1. Client Interface

The `Client` interface is the core abstraction that all provider implementations must satisfy:

```go
type Client interface {
    ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error)
    StreamChatCompletion(ctx context.Context, req ChatRequest) (<-chan StreamEvent, error)
    GetModelInfo() ModelInfo
    Close() error
}
```

**Key Features:**

- Context-aware operations with cancellation support
- Both synchronous and streaming response patterns
- Provider-agnostic request/response structures
- Resource cleanup through the `Close()` method

### 2. Factory Pattern

The `Factory` provides centralized client creation with configuration management.

### 3. Multimodal Content System

The library supports multimodal content through a type-safe interface system:

```mermaid
classDiagram
    class MessageContent {
        <<interface>>
        +Type() MessageType
        +Validate() error
        +Size() int64
    }

    class TextContent {
        +Text string
        +Type() MessageType
        +Validate() error
        +Size() int64
        +GetText() string
    }

    class ImageContent {
        +URL string
        +Data []byte
        +MimeType string
        +Type() MessageType
        +Validate() error
        +Size() int64
    }

    class FileContent {
        +URL string
        +Data []byte
        +Filename string
        +MimeType string
        +Type() MessageType
        +Validate() error
        +Size() int64
    }

    MessageContent <|-- TextContent
    MessageContent <|-- ImageContent
    MessageContent <|-- FileContent

    class Message {
        +Role MessageRole
        +Content []MessageContent
        +ToolCalls []ToolCall
        +Metadata map[string]any
        +AddContent(MessageContent)
        +GetContentByType(MessageType) []MessageContent
    }

    Message --> MessageContent
```

### 4. Streaming Architecture

The streaming architecture is a cornerstone of the go-llm library, designed to provide real-time, low-latency responses for interactive applications. The system uses a channel-based event-driven architecture that maintains thread safety while allowing for efficient resource management.

#### Core Streaming Components

The streaming system consists of several key components that work together to provide a seamless streaming experience:

1. **Stream Channel**: A buffered Go channel (`<-chan StreamEvent`) that delivers events asynchronously
2. **Event Types**: Standardized event structures for different phases of the streaming process
3. **Stream Processor**: Internal component that handles provider-specific parsing and normalization
4. **Context Management**: Full support for Go's context package for cancellation and timeouts

#### Streaming Flow Diagram

```mermaid
sequenceDiagram
    participant Client
    participant LLMClient
    participant Provider
    participant EventChannel
    participant StreamProcessor

    Client->>LLMClient: StreamChatCompletion(ctx, req)
    LLMClient->>Provider: Create Stream Request
    Provider-->>LLMClient: HTTP Stream
    LLMClient->>EventChannel: Create Channel
    LLMClient-->>Client: Return Channel

    loop Stream Processing
        Provider-->>LLMClient: Stream Chunk
        LLMClient->>StreamProcessor: Parse Chunk
        StreamProcessor->>EventChannel: Delta Event
    end

    Provider-->>LLMClient: Stream End
    LLMClient->>EventChannel: Done Event
    LLMClient->>EventChannel: Close Channel

    Note over Client, StreamProcessor: Events: Delta, Done, Error
```

#### Stream Event Types and Structure

The streaming system uses three primary event types, each serving a specific purpose in the communication flow:

**Delta Events**: Incremental content updates

- Contain partial message content that gets accumulated by the client
- Support multimodal content (text, images, files) within a single stream
- Include tool call deltas for function calling scenarios
- Provide real-time updates as the LLM generates content

**Done Events**: Stream completion with finish reason

- Signal the end of the streaming response
- Include metadata about why the stream ended (`stop`, `length`, `tool_calls`, etc.)
- Provide final response statistics and usage information
- Trigger cleanup operations in the client

**Error Events**: Error handling and recovery

- Contain standardized error information across all providers
- Support retry logic and graceful degradation
- Include provider-specific error codes and messages
- Allow for partial response recovery in case of interruptions

#### Event Processing Architecture

```mermaid
graph TD
    subgraph "Stream Event Processing"
        RawChunk[Raw Stream Chunk] --> Parser[Provider-Specific Parser]
        Parser --> Normalizer[Event Normalizer]
        Normalizer --> Validator[Event Validator]
        Validator --> EventBus[Event Channel]
    end

    subgraph "Client Side Processing"
        EventBus --> EventLoop[Client Event Loop]
        EventLoop --> |Delta| Accumulator[Content Accumulator]
        EventLoop --> |Done| Finalizer[Response Finalizer]
        EventLoop --> |Error| ErrorHandler[Error Handler]
    end

    subgraph "Content Assembly"
        Accumulator --> TextBuffer[Text Buffer]
        Accumulator --> ToolCalls[Tool Call Buffer]
        Accumulator --> Metadata[Metadata Buffer]
        TextBuffer --> FinalMessage[Complete Message]
        ToolCalls --> FinalMessage
        Metadata --> FinalMessage
    end

    style RawChunk fill:#e3f2fd
    style EventBus fill:#fff8e1
    style FinalMessage fill:#f1f8e9
```

#### Buffering and Flow Control

The streaming architecture implements intelligent buffering to balance performance with memory usage:

- **Channel Buffering**: Event channels use a configurable buffer size (default: 10 events) to prevent blocking
- **Backpressure Handling**: Automatic flow control when clients can't keep up with the stream
- **Memory Management**: Incremental content assembly with configurable size limits
- **Timeout Management**: Context-based timeouts for both connection and individual chunks

#### Thread Safety and Concurrency

The streaming system is designed for safe concurrent usage:

- **Goroutine Safety**: Each stream runs in its own goroutine with proper cleanup
- **Channel Safety**: Go channels provide built-in synchronization for event delivery
- **Resource Cleanup**: Automatic cleanup of HTTP connections and goroutines on context cancellation
- **Provider Isolation**: Each provider's streaming implementation is isolated to prevent cross-contamination

#### Provider-Specific Adaptations

While the streaming interface is uniform, each provider requires specific handling:

**OpenAI Streaming**:

- Uses Server-Sent Events (SSE) format
- Handles both text and tool call streaming
- Supports vision model streaming for multimodal content

**Ollama Streaming**:

- Uses newline-delimited JSON format
- Handles local model response patterns
- Supports custom model streaming capabilities

**Gemini Streaming**:

- Uses Google's streaming protocol
- Handles safety filter events
- Supports multimodal streaming with safety checks

#### Error Recovery and Resilience

The streaming architecture includes comprehensive error handling:

```mermaid
graph LR
    subgraph "Error Scenarios"
        NetworkError[Network Interruption]
        ParseError[Parse Error]
        ContextCancel[Context Cancellation]
        ProviderError[Provider Error]
    end

    subgraph "Recovery Strategies"
        NetworkError --> Retry[Automatic Retry]
        ParseError --> Skip[Skip Invalid Chunk]
        ContextCancel --> Cleanup[Graceful Cleanup]
        ProviderError --> Fallback[Error Event]
    end

    subgraph "Client Actions"
        Retry --> Resume[Resume Stream]
        Skip --> Continue[Continue Processing]
        Cleanup --> Return[Return to Client]
        Fallback --> Handle[Handle in Client]
    end

    style NetworkError fill:#ffebee
    style ParseError fill:#ffebee
    style Retry fill:#e8f5e8
    style Continue fill:#e8f5e8
```

#### Performance Characteristics

The streaming architecture is optimized for performance:

- **Low Latency**: Events are delivered as soon as they're parsed, typically within milliseconds
- **Memory Efficient**: Incremental processing prevents large memory allocations
- **Scalable**: Can handle hundreds of concurrent streams with proper resource management
- **Responsive**: Context cancellation propagates immediately through the entire pipeline

### 5. Message Router System

The `MessageRouter` provides flexible content processing through a handler-based architecture:

```mermaid
graph TB
    subgraph "Message Router"
        Router[MessageRouter] --> Registry[Handler Registry]
        Registry --> TextHandlers[Text Handlers]
        Registry --> ImageHandlers[Image Handlers]
        Registry --> FileHandlers[File Handlers]
        Registry --> DefaultHandler[Default Handler]
    end

    subgraph "Processing Flow"
        Message[Input Message] --> Router
        Router --> |Route by Type| HandlerChain[Handler Chain]
        HandlerChain --> ProcessedContent[Processed Content]
        ProcessedContent --> Response[Response Message]
    end

    subgraph "Handler Interface"
        Handler[MessageHandler Interface]
        Handler --> CanHandle[CanHandle Method]
        Handler --> HandleMethod[Handle Method]
    end

    style Router fill:#fff3e0
    style Message fill:#e3f2fd
    style Response fill:#f1f8e9
```

## ChatRequest Construction

Building `ChatRequest` objects is the primary way clients interact with the library:

### Basic Text Request

```go
request := llm.ChatRequest{
    Model: "gpt-4",
    Messages: []llm.Message{
        llm.NewTextMessage(llm.RoleUser, "Hello, how are you?"),
    },
    Temperature: &temperature, // *float32
    MaxTokens:   &maxTokens,   // *int
}
```

### Multimodal Request

```go
request := llm.ChatRequest{
    Model: "gpt-4-vision-preview",
    Messages: []llm.Message{
        {
            Role: llm.RoleUser,
            Content: []llm.MessageContent{
                llm.NewTextContent("What's in this image?"),
                llm.NewImageContentFromURL("https://example.com/image.jpg"),
            },
        },
    },
}
```

### Tool Calling Request

```go
request := llm.ChatRequest{
    Model: "gpt-4",
    Messages: []llm.Message{
        llm.NewTextMessage(llm.RoleUser, "What's the weather like?"),
    },
    Tools: []llm.Tool{
        {
            Type: "function",
            Function: llm.ToolFunction{
                Name:        "get_weather",
                Description: "Get weather information",
                Parameters: map[string]interface{}{
                    "type": "object",
                    "properties": map[string]interface{}{
                        "location": map[string]interface{}{
                            "type": "string",
                            "description": "City name",
                        },
                    },
                },
            },
        },
    },
}
```

## Streaming Support

### Event-Driven Streaming

The library provides comprehensive streaming support through an event-driven architecture:

```mermaid
graph LR
    subgraph "Stream Events"
        StreamEvent[StreamEvent] --> Delta[Delta Event]
        StreamEvent --> Done[Done Event]
        StreamEvent --> Error[Error Event]
    end

    subgraph "Delta Event Content"
        Delta --> StreamChoice[StreamChoice]
        StreamChoice --> MessageDelta[MessageDelta]
        MessageDelta --> ContentDeltas[Content Deltas]
        MessageDelta --> ToolCallDeltas[ToolCall Deltas]
    end

    subgraph "Event Processing"
        Client[Client Code] --> EventLoop[Event Loop]
        EventLoop --> |Delta| AccumulateContent[Accumulate Content]
        EventLoop --> |Done| FinalizeResponse[Finalize Response]
        EventLoop --> |Error| HandleError[Handle Error]
    end

    style StreamEvent fill:#fff8e1
    style Client fill:#e1f5fe
```

### Streaming Implementation Example

```go
stream, err := client.StreamChatCompletion(ctx, request)
if err != nil {
    return err
}

var fullResponse strings.Builder
for event := range stream {
    switch {
    case event.IsDelta():
        if len(event.Choice.Delta.Content) > 0 {
            if textContent, ok := event.Choice.Delta.Content[0].(*llm.TextContent); ok {
                fullResponse.WriteString(textContent.GetText())
                fmt.Print(textContent.GetText()) // Real-time output
            }
        }
    case event.IsDone():
        fmt.Println("\nStream complete:", event.Choice.FinishReason)
    case event.IsError():
        return fmt.Errorf("stream error: %v", event.Error)
    }
}
```

## Provider Integration

### Provider Implementation Pattern

Each provider follows a consistent implementation pattern:

```mermaid
graph TB
    subgraph "Provider Implementation"
        ProviderClient[Provider Client] --> Configuration[Configuration Setup]
        ProviderClient --> RequestConversion[Request Conversion]
        ProviderClient --> APICall[API Communication]
        ProviderClient --> ResponseConversion[Response Conversion]
        ProviderClient --> ErrorHandling[Error Handling]
    end

    subgraph "Common Capabilities"
        RequestConversion --> ModelSelection[Auto Model Selection]
        RequestConversion --> ContentTransformation[Content Transformation]
        RequestConversion --> ParameterMapping[Parameter Mapping]
    end

    subgraph "Provider Specific"
        APICall --> HTTPClient[HTTP Client]
        APICall --> Authentication[Authentication]
        APICall --> RateLimiting[Rate Limiting]
        APICall --> RetryLogic[Retry Logic]
    end

    style ProviderClient fill:#e8f5e8
    style Configuration fill:#f3e5f5
    style APICall fill:#e1f5fe
```

### Model Capability Registry

The library includes a comprehensive model registry that tracks capabilities:

```go
type ModelCapabilities struct {
    MaxTokens         int  `json:"max_tokens"`
    SupportsTools     bool `json:"supports_tools"`
    SupportsVision    bool `json:"supports_vision"`
    SupportsFiles     bool `json:"supports_files"`
    SupportsStreaming bool `json:"supports_streaming"`
}
```

## Security and Validation

### Content Validation Pipeline

```mermaid
graph TD
    subgraph "Validation Pipeline"
        Input[Content Input] --> SizeCheck[Size Validation]
        SizeCheck --> TypeCheck[Type Validation]
        TypeCheck --> ContentCheck[Content Validation]
        ContentCheck --> SecurityCheck[Security Validation]
        SecurityCheck --> Output[Validated Content]
    end

    subgraph "Security Measures"
        SecurityCheck --> URLValidation[URL Validation]
        SecurityCheck --> FileTypeCheck[File Type Check]
        SecurityCheck --> ContentSanitization[Content Sanitization]
        SecurityCheck --> SizeLimit[Size Limit Enforcement]
    end

    style Input fill:#e3f2fd
    style Output fill:#f1f8e9
    style SecurityCheck fill:#ffebee
```

### Error Handling

The library provides standardized error handling across all providers:

```go
type Error struct {
    Code       string `json:"code"`
    Message    string `json:"message"`
    Type       string `json:"type"`
    StatusCode int    `json:"status_code,omitempty"`
}
```

## Performance Considerations

### Concurrency Model

```mermaid
graph TB
    subgraph "Concurrent Operations"
        Request[Request] --> ClientPool[Client Pool]
        ClientPool --> Worker1[Worker 1]
        ClientPool --> Worker2[Worker 2]
        ClientPool --> WorkerN[Worker N]
    end

    subgraph "Resource Management"
        Worker1 --> Connection1[HTTP Connection]
        Worker2 --> Connection2[HTTP Connection]
        WorkerN --> ConnectionN[HTTP Connection]
    end

    subgraph "Thread Safety"
        MessageRouter --> RWMutex[RW Mutex]
        ContentHandlers --> HandlerMutex[Handler Mutex]
        StreamEvents --> ChannelSafety[Channel Safety]
    end

    style Request fill:#e3f2fd
    style ClientPool fill:#e8f5e8
    style RWMutex fill:#fff3e0
```

## Extension Points

### Custom Content Types

Developers can extend the system with custom content types:

```go
type CustomContent struct {
    Data []byte
    // Custom fields
}

func (c *CustomContent) Type() MessageType {
    return MessageType("custom")
}

func (c *CustomContent) Validate() error {
    // Custom validation logic
}

func (c *CustomContent) Size() int64 {
    return int64(len(c.Data))
}
```

### Custom Handlers

```go
handler := llm.NewTypedMessageHandler(
    llm.MessageTypeText,
    func(ctx context.Context, content llm.MessageContent) (llm.MessageContent, error) {
        // Custom processing logic
        return processedContent, nil
    },
)

router.RegisterHandler(llm.MessageTypeText, handler)
```

## Best Practices

### Request Construction

1. **Use appropriate models for content types** (vision models for images)
2. **Validate content before sending requests**
3. **Set reasonable timeouts and token limits**
4. **Handle errors gracefully with proper context**

### Streaming Usage

1. **Always handle all event types** (Delta, Done, Error)
2. **Implement proper cleanup** for channels and resources
3. **Use context for cancellation** in long-running streams
4. **Buffer content appropriately** for performance

### Provider Selection

1. **Choose providers based on specific needs** (local vs. cloud)
2. **Consider model capabilities** when building requests
3. **Implement fallback mechanisms** for critical applications
4. **Monitor usage and costs** for cloud providers

### Resource Management

1. **Always call Close()** on clients when done
2. **Use connection pooling** for high-throughput applications
3. **Implement proper retry logic** with exponential backoff
4. **Monitor memory usage** with large multimodal content

## Future Enhancements

The architecture is designed to support future enhancements:

- **Additional content types** (audio, video)
- **Custom provider implementations**
- **Advanced routing strategies**
- **Caching and persistence layers**
- **Metrics and observability**
- **Configuration management**

This architecture provides a solid foundation for building sophisticated LLM-powered applications while maintaining flexibility and extensibility.
