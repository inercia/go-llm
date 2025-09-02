# Mock Provider

The Mock Provider is a comprehensive testing utility that implements the LLM Client interface for testing and development purposes. It allows you to simulate various LLM behaviors without making actual API calls.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Features](#features)
- [Configuration](#configuration)
- [Testing Patterns](#testing-patterns)
- [Advanced Scenarios](#advanced-scenarios)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Basic Usage

### Creating a Mock Client

```go
import "github.com/inercia/go-llm/pkg/llm"

// Create basic mock client
mockClient := llm.NewMockClient("gpt-4", "mock")

// Use it like any other LLM client
response, err := mockClient.ChatCompletion(context.Background(), llm.ChatRequest{
    Model: "gpt-4",
    Messages: []llm.Message{
        {Role: llm.RoleUser, Content: []llm.MessageContent{llm.NewTextContent("Hello")}},
    },
})
```

### Simple Response Configuration

```go
mockClient := llm.NewMockClient("gpt-4", "mock")
    .WithSimpleResponse("Hello! How can I help you?")
    .WithSimpleResponse("I'm here to assist with your questions.")

// First call returns first response, second call returns second response
```

## Features

### ✅ Full LLM Interface Support

- Chat completion with configurable responses
- Streaming chat completion with realistic chunking
- Model information and capabilities
- Error simulation and handling

### ✅ Advanced Testing Capabilities

- Pre-configured responses and errors
- Tool calling simulation
- Multi-turn conversation support
- Latency and failure rate simulation
- Request logging and assertions

### ✅ Streaming Support

- Word-by-word streaming simulation
- Tool call streaming
- Custom stream events
- Context cancellation support

### ✅ Tool Integration

- Automatic tool call detection
- Custom tool call handlers
- Tool response simulation
- Function calling workflows

## Configuration

### Model Capabilities

```go
mockClient := llm.NewMockClient("custom-model", "mock")
    .WithModelCapabilities(
        8192,  // maxTokens
        true,  // supportsTools
        true,  // supportsVision
        false, // supportsFiles
        true,  // supportsStreaming
    )
```

### Latency Simulation

```go
mockClient := llm.NewMockClient("gpt-4", "mock")
    .WithLatency(200 * time.Millisecond) // Simulate 200ms response time
```

### Failure Rate Simulation

```go
mockClient := llm.NewMockClient("gpt-4", "mock")
    .WithFailureRate(0.1) // 10% chance of random failures
```

### Conversation State

```go
mockClient := llm.NewMockClient("gpt-4", "mock")
    .WithConversationState("user_preferences", map[string]string{
        "style": "detailed",
        "language": "english",
    })
```

## Testing Patterns

### 1. Basic Request-Response Testing

```go
func TestBasicInteraction(t *testing.T) {
    mockLLM := llm.NewMockClient("gpt-4", "mock")
        .WithSimpleResponse("Hello! How can I help you?")

    response, err := mockLLM.ChatCompletion(context.Background(), llm.ChatRequest{
        Model: "gpt-4",
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: []llm.MessageContent{llm.NewTextContent("Hello")}},
        },
    })

    assert.NoError(t, err)
    assert.Equal(t, "Hello! How can I help you?", getTextContent(response.Choices[0].Message.Content[0]))
}

func getTextContent(content llm.MessageContent) string {
    if textContent, ok := content.(*llm.TextContent); ok {
        return textContent.Text
    }
    return ""
}
```

### 2. Tool Calling Workflows

```go
func TestToolCallingWorkflow(t *testing.T) {
    mockLLM := llm.NewMockClient("gpt-4", "mock")
        .WithToolCall("web_search", map[string]interface{}{
            "query": "Go programming best practices",
        })
        .WithFunctionResult("web_search", "Go emphasizes simplicity, readability, and performance")

    // First call - should trigger tool call
    resp1, err := mockLLM.ChatCompletion(context.Background(), llm.ChatRequest{
        Model: "gpt-4",
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: []llm.MessageContent{llm.NewTextContent("Search for Go programming tips")}},
        },
    })

    assert.NoError(t, err)
    assert.Equal(t, "tool_calls", resp1.Choices[0].FinishReason)
    assert.Len(t, resp1.Choices[0].Message.ToolCalls, 1)
    assert.Equal(t, "web_search", resp1.Choices[0].Message.ToolCalls[0].Function.Name)

    // Second call - with tool result
    messages := []llm.Message{
        {Role: llm.RoleUser, Content: []llm.MessageContent{llm.NewTextContent("Search for Go programming tips")}},
        resp1.Choices[0].Message,
        {Role: llm.RoleTool, Content: []llm.MessageContent{llm.NewTextContent("Go emphasizes simplicity, readability, and performance")}},
    }

    resp2, err := mockLLM.ChatCompletion(context.Background(), llm.ChatRequest{
        Model:    "gpt-4",
        Messages: messages,
    })

    assert.NoError(t, err)
    assert.Contains(t, getTextContent(resp2.Choices[0].Message.Content[0]), "simplicity")
}
```

### 3. Multi-turn Conversations

```go
func TestMultiTurnConversation(t *testing.T) {
    exchanges := []llm.ConversationExchange{
        {Response: "I'd be happy to help you learn Go programming!"},
        {Response: "Go is known for its simplicity and strong concurrency support."},
        {ToolCall: &llm.MockToolCall{
            Name: "get_examples",
            Arguments: map[string]interface{}{"topic": "go-routines"},
        }},
        {Response: "Here are some practical examples of goroutines..."},
    }

    mockLLM := llm.NewMockClient("gpt-4", "mock")
        .WithConversation(exchanges)

    messages := []llm.Message{
        {Role: llm.RoleUser, Content: []llm.MessageContent{llm.NewTextContent("Help me learn Go")}},
    }

    // Simulate conversation turns
    for i := 0; i < len(exchanges); i++ {
        resp, err := mockLLM.ChatCompletion(context.Background(), llm.ChatRequest{
            Model:    "gpt-4",
            Messages: messages,
        })

        assert.NoError(t, err)
        messages = append(messages, resp.Choices[0].Message)
    }

    assert.True(t, mockLLM.AssertCallCount(4))
}
```

### 4. Streaming Responses

```go
func TestStreamingResponse(t *testing.T) {
    // Pre-configured streaming response
    streamEvents := llm.CreateWordByWordStream(
        "This is a streaming response that demonstrates chunked delivery",
        50*time.Millisecond,
    )

    mockLLM := llm.NewMockClient("gpt-4", "mock")
        .WithStreamResponse(streamEvents)

    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    stream, err := mockLLM.StreamChatCompletion(ctx, llm.ChatRequest{
        Model: "gpt-4",
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: []llm.MessageContent{llm.NewTextContent("Tell me something")}},
        },
    })

    assert.NoError(t, err)

    var collectedText strings.Builder
    var doneReceived bool

    for event := range stream {
        switch {
        case event.IsDelta():
            if event.Choice.Delta.Content != nil && len(event.Choice.Delta.Content) > 0 {
                if textContent, ok := event.Choice.Delta.Content[0].(*llm.TextContent); ok {
                    collectedText.WriteString(textContent.Text)
                }
            }
        case event.IsDone():
            doneReceived = true
        }
    }

    assert.True(t, doneReceived)
    assert.Contains(t, collectedText.String(), "streaming response")
}
```

### 5. Error Handling

```go
func TestErrorHandling(t *testing.T) {
    mockLLM := llm.NewMockClient("gpt-4", "mock")
        .WithError("rate_limit_exceeded", "Too many requests", "api_error")
        .WithError("model_overloaded", "Model temporarily unavailable", "service_error")
        .WithSimpleResponse("Success after retries")

    // First two calls should fail
    for i := 0; i < 2; i++ {
        _, err := mockLLM.ChatCompletion(context.Background(), llm.ChatRequest{
            Model: "gpt-4",
            Messages: []llm.Message{
                {Role: llm.RoleUser, Content: []llm.MessageContent{llm.NewTextContent("Hello")}},
            },
        })
        assert.Error(t, err)

        var llmErr *llm.Error
        assert.True(t, errors.As(err, &llmErr))
    }

    // Third call should succeed
    resp, err := mockLLM.ChatCompletion(context.Background(), llm.ChatRequest{
        Model: "gpt-4",
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: []llm.MessageContent{llm.NewTextContent("Hello")}},
        },
    })

    assert.NoError(t, err)
    assert.Equal(t, "Success after retries", getTextContent(resp.Choices[0].Message.Content[0]))
}
```

## Advanced Scenarios

### Testing Agent Behaviors

```go
func TestAgentWithComplexScenario(t *testing.T) {
    mockLLM := llm.NewMockClient("gpt-4", "mock")
        .WithLatency(100 * time.Millisecond)
        .WithSimpleResponse("I'll help you analyze this data.")
        .WithToolCall("fetch_data", map[string]interface{}{
            "source": "database",
            "query": "SELECT * FROM users WHERE active = true",
        })
        .WithSimpleResponse("Based on the data analysis, here are the insights...")

    // Test agent that uses LLM for data analysis
    agent := &DataAnalysisAgent{llm: mockLLM}
    result, err := agent.AnalyzeUserData("Show me active user statistics")

    assert.NoError(t, err)
    assert.Contains(t, result, "insights")

    // Verify LLM interactions
    assert.True(t, mockLLM.AssertCallCount(3))
    assert.True(t, mockLLM.AssertToolWasCalled("fetch_data"))
}
```

### Custom Tool Handlers

```go
func TestCustomToolHandlers(t *testing.T) {
    calculator := func(args string) (string, error) {
        var params map[string]interface{}
        json.Unmarshal([]byte(args), &params)
        expression, ok := params["expression"].(string)
        if !ok {
            return "", errors.New("invalid expression")
        }
        // Simple calculator logic...
        return "42", nil
    }

    weatherService := func(args string) (string, error) {
        return `{"temperature": 72, "condition": "sunny"}`, nil
    }

    mockLLM := llm.NewMockClient("gpt-4", "mock")
        .WithToolCallHandler("calculator", calculator)
        .WithToolCallHandler("weather", weatherService)

    // Test that handlers are properly registered
    assert.NotNil(t, mockLLM.toolCallHandlers["calculator"])
    assert.NotNil(t, mockLLM.toolCallHandlers["weather"])
}
```

### Streaming with Tool Calls

```go
func TestStreamingWithToolCalls(t *testing.T) {
    toolCallStream := llm.CreateToolCallStream(
        "I need to search for that information",
        "web_search",
        map[string]interface{}{"query": "Go programming"},
    )

    mockLLM := llm.NewMockClient("gpt-4", "mock")
        .WithStreamResponse(toolCallStream)

    ctx := context.Background()
    stream, err := mockLLM.StreamChatCompletion(ctx, llm.ChatRequest{
        Model: "gpt-4",
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: []llm.MessageContent{llm.NewTextContent("Search for Go info")}},
        },
    })

    assert.NoError(t, err)

    var hasToolCall bool
    var finishReason string

    for event := range stream {
        if event.IsDelta() && event.Choice.Delta.ToolCalls != nil {
            hasToolCall = true
        }
        if event.IsDone() {
            finishReason = event.Choice.FinishReason
        }
    }

    assert.True(t, hasToolCall)
    assert.Equal(t, "tool_calls", finishReason)
}
```

## Best Practices

### 1. Test Structure

```go
func TestLLMFeature(t *testing.T) {
    // Arrange - Set up mock with expected behavior
    mockLLM := llm.NewMockClient("gpt-4", "mock")
        .WithSimpleResponse("Expected response")

    // Act - Execute the code under test
    result, err := yourFunction(mockLLM)

    // Assert - Verify expected outcomes
    assert.NoError(t, err)
    assert.Equal(t, "expected", result)

    // Verify LLM interactions
    assert.True(t, mockLLM.AssertCallCount(1))
}
```

### 2. Realistic Simulation

```go
// Good: Realistic response that matches actual LLM behavior
mockLLM := llm.NewMockClient("gpt-4", "mock")
    .WithLatency(150 * time.Millisecond) // Simulate realistic latency
    .WithFailureRate(0.02) // 2% failure rate like real APIs
    .WithSimpleResponse("I understand you're asking about Go programming. Let me help you with that.")

// Avoid: Unrealistic responses
mockLLM := llm.NewMockClient("gpt-4", "mock")
    .WithSimpleResponse("OK") // Too brief for actual LLM
```

### 3. Comprehensive Error Testing

```go
func TestErrorRecovery(t *testing.T) {
    mockLLM := llm.NewMockClient("gpt-4", "mock")
        .WithError("network_error", "Connection timeout", "network")
        .WithError("rate_limit", "Rate limit exceeded", "api_error")
        .WithSimpleResponse("Success after retries")

    // Test your error handling logic
    result, err := retryableFunction(mockLLM, 3)
    assert.NoError(t, err)
    assert.Equal(t, "Success after retries", result)
}
```

### 4. State Management

```go
func TestStatefulConversation(t *testing.T) {
    mockLLM := llm.NewMockClient("gpt-4", "mock")
        .WithConversationState("user_context", map[string]interface{}{
            "name": "Alice",
            "preferences": []string{"detailed", "examples"},
        })

    // Your stateful logic can check conversation state
    // mockLLM.conversationState["user_context"]
}
```

### 5. Assertion Helpers

```go
func TestComplexInteraction(t *testing.T) {
    mockLLM := llm.NewMockClient("gpt-4", "mock")
        .WithSimpleResponse("I'll help you with that")
        .WithToolCall("helper_tool", map[string]interface{}{"param": "value"})

    // ... perform operations ...

    // Use built-in assertions
    assert.True(t, mockLLM.AssertCallCount(2))
    assert.True(t, mockLLM.AssertLastMessageContains("help"))
    assert.True(t, mockLLM.AssertToolWasCalled("helper_tool"))
}
```

## API Reference

### Constructor

#### `NewMockClient(modelName, provider string) *MockClient`

Creates a new mock LLM client with default capabilities.

### Configuration Methods

#### `WithSimpleResponse(content string) *MockClient`

Adds a simple text response to the response queue.

#### `WithToolCall(toolName string, args map[string]interface{}) *MockClient`

Adds a response that includes a tool call.

#### `WithError(code, message, errorType string) *MockClient`

Adds an error to be returned by subsequent calls.

#### `WithLatency(duration time.Duration) *MockClient`

Configures simulated latency for all requests.

#### `WithFailureRate(rate float64) *MockClient`

Sets random failure simulation rate (0.0 to 1.0).

#### `WithModelCapabilities(maxTokens int, supportsTools, supportsVision, supportsFiles, supportsStreaming bool) *MockClient`

Configures the model's reported capabilities.

#### `WithConversationState(key string, value interface{}) *MockClient`

Sets conversation state for context-aware responses.

#### `WithStreamResponse(events []StreamEvent) *MockClient`

Adds a pre-configured streaming response.

#### `WithToolCallHandler(toolName string, handler func(args string) (string, error)) *MockClient`

Registers a custom handler for specific tool calls.

### Conversation Helpers

#### `WithConversation(exchanges []ConversationExchange) *MockClient`

Sets up a multi-turn conversation scenario.

#### `WithSystemMessage(content string) *MockClient`

Creates a response as if from a system message.

#### `WithFunctionResult(functionName, result string) *MockClient`

Creates a response that follows a function call.

#### `WithMultiStepResponse(steps []string) *MockClient`

Creates a complex response with reasoning steps.

### Streaming Helpers

#### `CreateWordByWordStream(text string, delay time.Duration) []StreamEvent`

Creates a streaming response that sends words individually.

#### `CreateToolCallStream(initialText, toolName string, args map[string]interface{}) []StreamEvent`

Creates a streaming response that includes a tool call.

### Test Assertion Methods

#### `AssertCallCount(expected int) bool`

Verifies the number of calls made to the mock.

#### `AssertLastMessageContains(text string) bool`

Checks if the last user message contains specific text.

#### `AssertToolWasCalled(toolName string) bool`

Checks if a specific tool was called in any request.

### Data Access Methods

#### `GetCallLog() []ChatRequest`

Returns all requests made to this mock client.

#### `GetLastCall() *ChatRequest`

Returns the most recent request made to this mock client.

#### `Reset() *MockClient`

Clears all responses, errors, and call logs.

### Core Interface Methods

#### `ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error)`

Implements the standard chat completion interface.

#### `StreamChatCompletion(ctx context.Context, req ChatRequest) (<-chan StreamEvent, error)`

Implements the streaming chat completion interface.

#### `GetModelInfo() ModelInfo`

Returns the configured model information.

#### `Close() error`

Implements the close method (no-op for mock).

## Integration Examples

See the extensive examples in the mock.go file for detailed usage patterns including:

- Basic request-response testing
- Tool calling workflows
- Multi-turn conversations
- Streaming responses
- Error handling scenarios
- Advanced configuration
- Custom tool handlers
- Agent testing patterns

The Mock Provider is designed to make testing LLM-powered applications straightforward and reliable, enabling you to focus on your application logic rather than external API dependencies.

## Quick Start Example

Here's a complete example showing how to test an AI agent using the Mock Provider:

```go
package main

import (
    "context"
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/inercia/go-llm/pkg/llm"
)

// Example AI agent that uses LLM for research
type ResearchAgent struct {
    llm llm.Client
}

func (a *ResearchAgent) Research(topic string) (string, error) {
    response, err := a.llm.ChatCompletion(context.Background(), llm.ChatRequest{
        Model: "gpt-4",
        Messages: []llm.Message{
            {
                Role: llm.RoleUser,
                Content: []llm.MessageContent{
                    llm.NewTextContent("Research the topic: " + topic),
                },
            },
        },
    })

    if err != nil {
        return "", err
    }

    if textContent, ok := response.Choices[0].Message.Content[0].(*llm.TextContent); ok {
        return textContent.Text, nil
    }

    return "", fmt.Errorf("unexpected content type")
}

// Test the agent using Mock Provider
func TestResearchAgent(t *testing.T) {
    // Setup mock LLM with realistic behavior
    mockLLM := llm.NewMockClient("gpt-4", "mock")
        .WithLatency(100 * time.Millisecond)  // Realistic response time
        .WithSimpleResponse("Based on my research, Go is a programming language developed by Google that emphasizes simplicity, performance, and strong concurrency support through goroutines.")

    // Create agent with mock LLM
    agent := &ResearchAgent{llm: mockLLM}

    // Test the research functionality
    result, err := agent.Research("Go programming language")

    // Verify results
    assert.NoError(t, err)
    assert.Contains(t, result, "Go")
    assert.Contains(t, result, "Google")
    assert.Contains(t, result, "goroutines")

    // Verify LLM interactions
    assert.True(t, mockLLM.AssertCallCount(1))
    assert.True(t, mockLLM.AssertLastMessageContains("Go programming language"))
}
```

This example demonstrates:

- ✅ **Zero External Dependencies**: No API keys or network calls needed
- ✅ **Realistic Simulation**: Latency and response patterns match real LLMs
- ✅ **Comprehensive Testing**: Full verification of both logic and LLM interactions
- ✅ **Easy Maintenance**: Simple setup with fluent configuration API
- ✅ **Deterministic Results**: Consistent test outcomes for reliable CI/CD

The Mock Provider transforms LLM testing from a complex integration challenge into straightforward unit testing.
