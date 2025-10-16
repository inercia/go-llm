# Tools and Function Calling

This guide covers how to use tools and function calling with the Go LLM library. Tools enable LLMs to interact with external systems, perform calculations, and access real-time information.

## Overview

Tools (also known as function calling) allow language models to:

- Call external APIs and services
- Perform calculations and data processing
- Access databases and file systems
- Execute custom business logic
- Interact with other systems in real-time

The library supports OpenAI-compatible tool definitions and provides both synchronous and streaming tool execution.

## Basic Tool Usage

### Simple Calculator Tool

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    "github.com/inercia/go-llm/pkg/llm"
    "github.com/inercia/go-llm/pkg/factory"

)

func main() {
    // Create client - registry auto-imports all providers
    factory := factory.New()

    // Create client
    client, err := factory.CreateClient(llm.ClientConfig{
        Provider: "openai",
        Model:    "gpt-3.5-turbo",
        APIKey:   "your-api-key",
    })
    if err != nil {
        log.Fatal("Failed to create client:", err)
    }
    defer client.Close()

    // Define a calculator tool
    calculatorTool := llm.Tool{
        Type: "function",
        Function: llm.ToolFunction{
            Name:        "calculate",
            Description: "Perform basic arithmetic operations",
            Parameters: map[string]interface{}{
                "type": "object",
                "properties": map[string]interface{}{
                    "operation": map[string]interface{}{
                        "type":        "string",
                        "description": "The operation to perform",
                        "enum":        []string{"add", "subtract", "multiply", "divide"},
                    },
                    "a": map[string]interface{}{
                        "type":        "number",
                        "description": "First number",
                    },
                    "b": map[string]interface{}{
                        "type":        "number",
                        "description": "Second number",
                    },
                },
                "required": []string{"operation", "a", "b"},
            },
        },
    }

    // Create request with tool
    req := llm.ChatRequest{
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: []llm.MessageContent{
                llm.NewTextContent("Calculate 25 * 4 using the calculator tool."),
            }},
        },
        Tools: []llm.Tool{calculatorTool},
    }

    // Send request
    resp, err := client.ChatCompletion(context.Background(), req)
    if err != nil {
        log.Fatal("Request failed:", err)
    }

    // Check for tool calls
    if len(resp.Choices) > 0 && resp.Choices[0].Message.ToolCalls != nil {
        for _, toolCall := range resp.Choices[0].Message.ToolCalls {
            fmt.Printf("Tool called: %s\n", toolCall.Function.Name)
            fmt.Printf("Arguments: %s\n", toolCall.Function.Arguments)

            // Execute the tool
            result, err := executeCalculator(toolCall.Function.Arguments)
            if err != nil {
                log.Printf("Tool execution failed: %v", err)
                continue
            }

            fmt.Printf("Result: %v\n", result)
        }
    }

}

// Example tool implementation
func executeCalculator(arguments string) (interface{}, error) {
    var args struct {
        Operation string `json:"operation"`
        A float64 `json:"a"`
        B float64 `json:"b"`
    }

    if err := json.Unmarshal([]byte(arguments), &args); err != nil {
        return nil, fmt.Errorf("invalid arguments: %w", err)
    }

    switch args.Operation {
    case "add":
        return args.A + args.B, nil
    case "subtract":
        return args.A - args.B, nil
    case "multiply":
        return args.A * args.B, nil
    case "divide":
        if args.B == 0 {
            return nil, fmt.Errorf("division by zero")
        }
        return args.A / args.B, nil
    default:
        return nil, fmt.Errorf("unsupported operation: %s", args.Operation)
    }

}
```

## Tool Definition Structure

### Tool Schema

```go
type Tool struct {
    Type     string       `json:"type"`     // Always "function"
    Function ToolFunction `json:"function"` // Function definition
}

type ToolFunction struct {
    Name        string                 `json:"name"`        // Function name (required)
    Description string                 `json:"description"` // What the function does (required)
    Parameters  map[string]interface{} `json:"parameters"`  // JSON Schema for parameters
}
````

### Parameter Schema Examples

#### Simple Text Input

```go
Parameters: map[string]interface{}{
    "type": "object",
    "properties": map[string]interface{}{
        "query": map[string]interface{}{
            "type":        "string",
            "description": "Search query to execute",
        },
    },
    "required": []string{"query"},
}
```

#### Multiple Parameter Types

```go
Parameters: map[string]interface{}{
    "type": "object",
    "properties": map[string]interface{}{
        "name": map[string]interface{}{
            "type":        "string",
            "description": "Person's name",
        },
        "age": map[string]interface{}{
            "type":        "integer",
            "description": "Person's age",
            "minimum":     0,
            "maximum":     150,
        },
        "active": map[string]interface{}{
            "type":        "boolean",
            "description": "Whether the person is active",
        },
        "category": map[string]interface{}{
            "type":        "string",
            "description": "Category selection",
            "enum":        []string{"personal", "business", "other"},
        },
    },
    "required": []string{"name", "age"},
}
```

#### Array and Object Parameters

```go
Parameters: map[string]interface{}{
    "type": "object",
    "properties": map[string]interface{}{
        "items": map[string]interface{}{
            "type":        "array",
            "description": "List of items to process",
            "items": map[string]interface{}{
                "type": "string",
            },
        },
        "config": map[string]interface{}{
            "type":        "object",
            "description": "Configuration options",
            "properties": map[string]interface{}{
                "timeout": map[string]interface{}{
                    "type":        "integer",
                    "description": "Timeout in seconds",
                },
                "retry": map[string]interface{}{
                    "type":        "boolean",
                    "description": "Whether to retry on failure",
                },
            },
        },
    },
    "required": []string{"items"},
}
```

## Advanced Tool Patterns

### Multiple Tools

```go
func createMultipleTools() []llm.Tool {
    return []llm.Tool{
        // Web search tool
        {
            Type: "function",
            Function: llm.ToolFunction{
                Name:        "web_search",
                Description: "Search the web for information",
                Parameters: map[string]interface{}{
                    "type": "object",
                    "properties": map[string]interface{}{
                        "query": map[string]interface{}{
                            "type":        "string",
                            "description": "Search query",
                        },
                        "num_results": map[string]interface{}{
                            "type":        "integer",
                            "description": "Number of results to return",
                            "default":     5,
                        },
                    },
                    "required": []string{"query"},
                },
            },
        },
        // Time tool
        {
            Type: "function",
            Function: llm.ToolFunction{
                Name:        "get_current_time",
                Description: "Get the current date and time",
                Parameters: map[string]interface{}{
                    "type": "object",
                    "properties": map[string]interface{}{
                        "timezone": map[string]interface{}{
                            "type":        "string",
                            "description": "Timezone (optional)",
                        },
                    },
                },
            },
        },
        // File operations tool
        {
            Type: "function",
            Function: llm.ToolFunction{
                Name:        "read_file",
                Description: "Read contents of a file",
                Parameters: map[string]interface{}{
                    "type": "object",
                    "properties": map[string]interface{}{
                        "filepath": map[string]interface{}{
                            "type":        "string",
                            "description": "Path to the file to read",
                        },
                        "encoding": map[string]interface{}{
                            "type":        "string",
                            "description": "File encoding",
                            "enum":        []string{"utf-8", "ascii", "latin1"},
                            "default":     "utf-8",
                        },
                    },
                    "required": []string{"filepath"},
                },
            },
        },
    }
}
```

### Complete Tool Conversation Flow

```go
func toolConversationFlow(client llm.Client) error {
    tools := createMultipleTools()

    // Initial request
    req := llm.ChatRequest{
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: []llm.MessageContent{
                llm.NewTextContent("What time is it and search for recent news about AI."),
            }},
        },
        Tools: tools,
    }

    ctx := context.Background()
    resp, err := client.ChatCompletion(ctx, req)
    if err != nil {
        return err
    }

    // Process tool calls
    if len(resp.Choices) == 0 || resp.Choices[0].Message.ToolCalls == nil {
        fmt.Println("No tool calls made")
        return nil
    }

    // Collect messages for conversation continuation
    messages := req.Messages
    messages = append(messages, resp.Choices[0].Message)

    // Execute each tool call
    for _, toolCall := range resp.Choices[0].Message.ToolCalls {
        result, err := executeToolCall(toolCall)
        if err != nil {
            result = fmt.Sprintf("Error: %v", err)
        }

        // Add tool result message
        toolMessage := llm.Message{
            Role: llm.RoleTool,
            Content: []llm.MessageContent{
                llm.NewTextContent(result),
            },
            ToolCallID: toolCall.ID,
        }
        messages = append(messages, toolMessage)
    }

    // Continue conversation with tool results
    followUpReq := llm.ChatRequest{
        Messages: messages,
        Tools:    tools,
    }

    finalResp, err := client.ChatCompletion(ctx, followUpReq)
    if err != nil {
        return err
    }

    fmt.Printf("Final response: %s\n", finalResp.Choices[0].Message.GetText())
    return nil
}

func executeToolCall(toolCall llm.ToolCall) (string, error) {
    switch toolCall.Function.Name {
    case "get_current_time":
        return executeTimeFunction(toolCall.Function.Arguments)
    case "web_search":
        return executeSearchFunction(toolCall.Function.Arguments)
    case "read_file":
        return executeFileFunction(toolCall.Function.Arguments)
    default:
        return "", fmt.Errorf("unknown tool: %s", toolCall.Function.Name)
    }
}
```

## Streaming with Tools

Tools can be used with streaming responses:

```go
func streamingWithTools(client llm.Client) error {
    weatherTool := llm.Tool{
        Type: "function",
        Function: llm.ToolFunction{
            Name:        "get_weather",
            Description: "Get weather information for a location",
            Parameters: map[string]interface{}{
                "type": "object",
                "properties": map[string]interface{}{
                    "location": map[string]interface{}{
                        "type":        "string",
                        "description": "City and country/state",
                    },
                    "units": map[string]interface{}{
                        "type":        "string",
                        "description": "Temperature units",
                        "enum":        []string{"celsius", "fahrenheit"},
                        "default":     "celsius",
                    },
                },
                "required": []string{"location"},
            },
        },
    }

    req := llm.ChatRequest{
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: []llm.MessageContent{
                llm.NewTextContent("What's the weather like in Tokyo?"),
            }},
        },
        Tools:  []llm.Tool{weatherTool},
        Stream: true,
    }

    stream, err := client.StreamChatCompletion(context.Background(), req)
    if err != nil {
        return err
    }

    var toolCalls []llm.ToolCall
    var textResponse strings.Builder

    for event := range stream {
        switch {
        case event.IsError():
            return fmt.Errorf("stream error: %s", event.Error.Message)

        case event.IsDelta():
            if event.Choice.Delta != nil {
                // Handle tool calls in stream
                if event.Choice.Delta.ToolCalls != nil {
                    for _, toolCall := range event.Choice.Delta.ToolCalls {
                        if toolCall.Function != nil && toolCall.Function.Name != "" {
                            fmt.Printf("Tool call detected: %s\n", toolCall.Function.Name)
                            toolCalls = append(toolCalls, toolCall)
                        }
                    }
                }

                // Handle text content
                for _, content := range event.Choice.Delta.Content {
                    if textContent, ok := content.(*llm.TextContent); ok {
                        chunk := textContent.GetText()
                        textResponse.WriteString(chunk)
                        fmt.Print(chunk)
                    }
                }
            }

        case event.IsDone():
            fmt.Printf("\nStream completed: %s\n", event.Choice.FinishReason)

            // Execute any tool calls
            for _, toolCall := range toolCalls {
                result, err := executeWeatherTool(toolCall.Function.Arguments)
                if err != nil {
                    fmt.Printf("Tool error: %v\n", err)
                } else {
                    fmt.Printf("Weather result: %s\n", result)
                }
            }
        }
    }

    return nil
}

func executeWeatherTool(arguments string) (string, error) {
    var args struct {
        Location string `json:"location"`
        Units    string `json:"units,omitempty"`
    }

    if err := json.Unmarshal([]byte(arguments), &args); err != nil {
        return "", err
    }

    // Mock weather data
    return fmt.Sprintf("Weather in %s: 22°C, partly cloudy", args.Location), nil
}
```

## Tool Best Practices

### 1. Clear Tool Descriptions

Write descriptive function names and descriptions:

```go
// Good
Function: llm.ToolFunction{
    Name:        "search_company_database",
    Description: "Search internal company database for employee information, projects, or organizational data",
    // ...
}

// Bad
Function: llm.ToolFunction{
    Name:        "search",
    Description: "Search for stuff",
    // ...
}
```

### 2. Comprehensive Parameter Validation

```go
func validateToolParameters(args map[string]interface{}) error {
    // Check required parameters
    if _, exists := args["query"]; !exists {
        return fmt.Errorf("missing required parameter: query")
    }

    // Validate parameter types
    query, ok := args["query"].(string)
    if !ok {
        return fmt.Errorf("query must be a string")
    }

    // Validate parameter values
    if len(strings.TrimSpace(query)) == 0 {
        return fmt.Errorf("query cannot be empty")
    }

    return nil
}
```

### 3. Error Handling in Tools

```go
func executeToolSafely(toolCall llm.ToolCall) string {
    defer func() {
        if r := recover(); r != nil {
            log.Printf("Tool execution panic: %v", r)
        }
    }()

    result, err := executeToolCall(toolCall)
    if err != nil {
        // Return structured error information
        return fmt.Sprintf("Tool execution failed: %v", err)
    }

    return result
}
```

### 4. Tool Response Formatting

Structure tool responses for better LLM understanding:

```go
func formatToolResponse(data interface{}, success bool, message string) string {
    response := map[string]interface{}{
        "success": success,
        "message": message,
        "data":    data,
    }

    jsonData, err := json.Marshal(response)
    if err != nil {
        return fmt.Sprintf(`{"success": false, "message": "JSON encoding failed", "data": null}`)
    }

    return string(jsonData)
}

// Usage
func searchDatabase(query string) string {
    results, err := performSearch(query)
    if err != nil {
        return formatToolResponse(nil, false, err.Error())
    }

    return formatToolResponse(results, true, "Search completed successfully")
}
```

### 5. Rate Limiting and Resource Management

```go
type ToolExecutor struct {
    rateLimiter *time.Ticker
    semaphore   chan struct{}
}

func NewToolExecutor(maxConcurrent int, rateLimit time.Duration) *ToolExecutor {
    return &ToolExecutor{
        rateLimiter: time.NewTicker(rateLimit),
        semaphore:   make(chan struct{}, maxConcurrent),
    }
}

func (te *ToolExecutor) Execute(toolCall llm.ToolCall) (string, error) {
    // Acquire semaphore for concurrency control
    te.semaphore <- struct{}{}
    defer func() { <-te.semaphore }()

    // Rate limiting
    <-te.rateLimiter.C

    // Execute tool
    return executeToolCall(toolCall)
}
```

## Common Tool Patterns

### 1. API Integration Tools

```go
func createAPITool(serviceName, endpoint string) llm.Tool {
    return llm.Tool{
        Type: "function",
        Function: llm.ToolFunction{
            Name:        fmt.Sprintf("call_%s_api", strings.ToLower(serviceName)),
            Description: fmt.Sprintf("Call %s API to retrieve data", serviceName),
            Parameters: map[string]interface{}{
                "type": "object",
                "properties": map[string]interface{}{
                    "endpoint": map[string]interface{}{
                        "type":        "string",
                        "description": "API endpoint path",
                    },
                    "params": map[string]interface{}{
                        "type":        "object",
                        "description": "Query parameters",
                    },
                },
                "required": []string{"endpoint"},
            },
        },
    }
}
```

### 2. Database Query Tools

```go
func createDatabaseTool() llm.Tool {
    return llm.Tool{
        Type: "function",
        Function: llm.ToolFunction{
            Name:        "query_database",
            Description: "Execute SQL queries on the application database",
            Parameters: map[string]interface{}{
                "type": "object",
                "properties": map[string]interface{}{
                    "table": map[string]interface{}{
                        "type":        "string",
                        "description": "Database table name",
                        "enum":        []string{"users", "orders", "products", "analytics"},
                    },
                    "filters": map[string]interface{}{
                        "type":        "object",
                        "description": "WHERE clause filters",
                    },
                    "limit": map[string]interface{}{
                        "type":        "integer",
                        "description": "Maximum number of results",
                        "default":     10,
                        "maximum":     100,
                    },
                },
                "required": []string{"table"},
            },
        },
    }
}
```

### 3. File System Tools

```go
func createFileSystemTools() []llm.Tool {
    return []llm.Tool{
        {
            Type: "function",
            Function: llm.ToolFunction{
                Name:        "list_directory",
                Description: "List files and directories in a path",
                Parameters: map[string]interface{}{
                    "type": "object",
                    "properties": map[string]interface{}{
                        "path": map[string]interface{}{
                            "type":        "string",
                            "description": "Directory path to list",
                        },
                        "include_hidden": map[string]interface{}{
                            "type":        "boolean",
                            "description": "Include hidden files",
                            "default":     false,
                        },
                    },
                    "required": []string{"path"},
                },
            },
        },
        {
            Type: "function",
            Function: llm.ToolFunction{
                Name:        "read_file_content",
                Description: "Read the contents of a text file",
                Parameters: map[string]interface{}{
                    "type": "object",
                    "properties": map[string]interface{}{
                        "filepath": map[string]interface{}{
                            "type":        "string",
                            "description": "Path to the file to read",
                        },
                        "max_lines": map[string]interface{}{
                            "type":        "integer",
                            "description": "Maximum lines to read",
                            "default":     100,
                        },
                    },
                    "required": []string{"filepath"},
                },
            },
        },
    }
}
```

## Error Handling and Debugging

### Tool Execution Monitoring

```go
func monitoredToolExecution(toolCall llm.ToolCall) (string, error) {
    start := time.Now()

    log.Printf("Executing tool: %s with args: %s",
               toolCall.Function.Name, toolCall.Function.Arguments)

    defer func() {
        duration := time.Since(start)
        log.Printf("Tool %s completed in %v", toolCall.Function.Name, duration)
    }()

    // Execute with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    resultChan := make(chan string, 1)
    errorChan := make(chan error, 1)

    go func() {
        result, err := executeToolCall(toolCall)
        if err != nil {
            errorChan <- err
        } else {
            resultChan <- result
        }
    }()

    select {
    case result := <-resultChan:
        return result, nil
    case err := <-errorChan:
        return "", fmt.Errorf("tool execution failed: %w", err)
    case <-ctx.Done():
        return "", fmt.Errorf("tool execution timeout: %s", toolCall.Function.Name)
    }
}
```

### Common Issues and Solutions

#### Issue: Tool Not Called

**Causes**: Unclear descriptions, missing required parameters, model limitations

**Solutions**:

- Make tool descriptions more specific and action-oriented
- Provide examples in the description
- Ensure parameter schemas are correct
- Check model compatibility with tools

#### Issue: Invalid Tool Arguments

**Causes**: Complex parameter schemas, model misunderstanding requirements

**Solutions**:

```go
func robustArgumentParsing(arguments string, target interface{}) error {
    // Try normal parsing first
    if err := json.Unmarshal([]byte(arguments), target); err == nil {
        return nil
    }

    // Try to clean and fix common issues
    cleaned := strings.TrimSpace(arguments)
    if !strings.HasPrefix(cleaned, "{") {
        cleaned = "{" + cleaned + "}"
    }

    // Attempt parsing again
    return json.Unmarshal([]byte(cleaned), target)
}
```

#### Issue: Tool Execution Failures

**Causes**: External service issues, network problems, resource constraints

**Solutions**: Implement retries with exponential backoff

```go
func executeWithRetry(toolCall llm.ToolCall, maxRetries int) (string, error) {
    var lastErr error

    for i := 0; i < maxRetries; i++ {
        result, err := executeToolCall(toolCall)
        if err == nil {
            return result, nil
        }

        lastErr = err
        backoff := time.Duration(math.Pow(2, float64(i))) * time.Second
        time.Sleep(backoff)
    }

    return "", fmt.Errorf("tool failed after %d retries: %w", maxRetries, lastErr)
}
```

## Performance Optimization

### 1. Tool Caching

Cache tool results for frequently called functions:

```go
type ToolCache struct {
    cache map[string]CacheEntry
    mutex sync.RWMutex
}

type CacheEntry struct {
    Result    string
    Timestamp time.Time
    TTL       time.Duration
}

func (tc *ToolCache) Get(key string) (string, bool) {
    tc.mutex.RLock()
    defer tc.mutex.RUnlock()

    entry, exists := tc.cache[key]
    if !exists || time.Since(entry.Timestamp) > entry.TTL {
        return "", false
    }

    return entry.Result, true
}
```

### 2. Parallel Tool Execution

Execute multiple independent tools concurrently:

```go
func executeToolsConcurrently(toolCalls []llm.ToolCall) []string {
    results := make([]string, len(toolCalls))
    var wg sync.WaitGroup

    for i, toolCall := range toolCalls {
        wg.Add(1)
        go func(index int, tc llm.ToolCall) {
            defer wg.Done()
            result, err := executeToolCall(tc)
            if err != nil {
                results[index] = fmt.Sprintf("Error: %v", err)
            } else {
                results[index] = result
            }
        }(i, toolCall)
    }

    wg.Wait()
    return results
}
```

## See Also

- [Streaming Documentation](streaming.md) - Using tools with streaming responses
- [Multimodal Documentation](multimodal.md) - Combining tools with images and files
- [Architecture Overview](architecture.md) - Understanding client architecture
- [Examples Directory](../examples/) - Complete runnable examples with tools
