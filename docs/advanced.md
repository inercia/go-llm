# Advanced Patterns

## Core Features

For detailed information on key library capabilities, see these dedicated guides:

- **[Streaming Chat Completion](streaming.md)**: Real-time response generation, event handling, and advanced streaming patterns
- **[Tools & Function Calling](tools.md)**: Tool definition, execution workflows, and integration patterns
- **[Multimodal Messages](multimodal.md)**: Working with images, files, and mixed content types

## Structured Outputs with JSON Schema

Get reliable, schema-validated JSON responses using OpenAI-style structured outputs that work across all providers. The library supports both basic JSON mode and strict JSON Schema validation generated from Go structs using the powerful [swaggest/jsonschema-go](https://github.com/swaggest/jsonschema-go) library.

```go
// Define your response structure
type Analysis struct {
    Sentiment  string   `json:"sentiment" description:"positive, negative, or neutral"`
    Keywords   []string `json:"keywords" description:"important keywords"`
    Confidence float64  `json:"confidence" minimum:"0" maximum:"1"`
}

// Generate strict JSON Schema from Go struct
responseFormat, err := llm.NewJSONSchemaResponseFormatStrictFromStruct(
    "analysis_result",
    "Text analysis with sentiment and keywords",
    Analysis{},
)

// Use with any provider - OpenAI gets native support, others use prompt engineering
resp, err := client.ChatCompletion(ctx, llm.ChatRequest{
    Model: "gpt-4o-2024-08-06",
    Messages: []llm.Message{
        llm.NewTextMessage(llm.RoleUser, "Analyze this text: 'I love this product!'"),
    },
    ResponseFormat: responseFormat, // Ensures structured JSON output
})

// Extract directly to struct with validation
var analysis Analysis
err = llm.ExtractAndValidateJSONToStruct(resp.Choices[0].Message.GetText(), &analysis, responseFormat.JSONSchema.Schema)
```

**Provider Support**: OpenAI/OpenRouter provide native JSON Schema support with strict validation, while Gemini/Ollama use intelligent prompt engineering to achieve structured outputs. Check `ModelInfo.SupportsJSONSchema` for native support availability.

## Model Information

Retrieve details about the current model:

```go
modelInfo := client.GetModelInfo()
fmt.Printf("Model: %s, Supports Streaming: %t, JSON Schema: %t\n",
    modelInfo.Name, modelInfo.SupportsStreaming, modelInfo.SupportsJSONSchema)
```

## Remote Provider Health Monitoring

Monitor the health and status of remote LLM providers using the `GetRemote()` method. This feature provides cached health checks to avoid excessive API calls while giving you real-time visibility into provider availability.

### Basic Health Check

```go
// Check provider health and status
remoteInfo := client.GetRemote()
fmt.Printf("Provider: %s\n", remoteInfo.Name)

if remoteInfo.Status != nil {
    if remoteInfo.Status.Healthy != nil {
        fmt.Printf("Healthy: %t\n", *remoteInfo.Status.Healthy)
    }

    if remoteInfo.Status.LastChecked != nil {
        fmt.Printf("Last Checked: %s\n", remoteInfo.Status.LastChecked.Format(time.RFC3339))
    }
}
```

### Health Check Caching

To prevent excessive API calls, health status is automatically cached for **5 minutes** (defined by `llm.DefaultHealthCheckInterval`). The cache works as follows:

- **First call**: Performs actual health check and caches the result
- **Subsequent calls**: Returns cached result if less than 5 minutes have passed
- **Cache expiry**: Automatically refreshes when 5+ minutes have elapsed

```go
// This will perform an actual health check
remoteInfo1 := client.GetRemote()
fmt.Printf("First check at: %s\n", remoteInfo1.Status.LastChecked.Format(time.RFC3339))

// This will use cached result (same timestamp)
remoteInfo2 := client.GetRemote()
fmt.Printf("Second check at: %s\n", remoteInfo2.Status.LastChecked.Format(time.RFC3339))

// After 5 minutes, this would perform a new health check
time.Sleep(6 * time.Minute)
remoteInfo3 := client.GetRemote()
// remoteInfo3.Status.LastChecked will be updated
```

### Provider-Specific Health Checks

Each provider implements lightweight health checks to minimize resource usage:

- **OpenAI/OpenRouter**: Lists available models (lightweight API call)
- **Gemini**: Creates a minimal chat session with 1 token output limit
- **DeepSeek**: Sends a minimal chat completion with 1 token limit
- **Ollama**: Queries the `/api/tags` endpoint (model listing)
- **Mock**: Always returns healthy (no actual remote check needed)

### Middleware Integration

The `GetRemote()` method works seamlessly through middleware:

```go
// Health checks work through middleware layers
enhancedClient := llm.ClientWithMiddleware(client, middlewares)
remoteInfo := enhancedClient.GetRemote() // Delegates to underlying client

// Useful for monitoring wrapped clients
fmt.Printf("Enhanced client provider: %s, healthy: %t\n",
    remoteInfo.Name, *remoteInfo.Status.Healthy)
```

### Production Monitoring

Use `GetRemote()` for production health monitoring and alerting:

```go
func checkProviderHealth(client llm.Client) error {
    remoteInfo := client.GetRemote()

    if remoteInfo.Status == nil || remoteInfo.Status.Healthy == nil {
        return fmt.Errorf("provider %s: health status unknown", remoteInfo.Name)
    }

    if !*remoteInfo.Status.Healthy {
        return fmt.Errorf("provider %s: unhealthy (last checked: %v)",
            remoteInfo.Name, remoteInfo.Status.LastChecked)
    }

    fmt.Printf("✅ Provider %s is healthy\n", remoteInfo.Name)
    return nil
}

// Monitor multiple providers
providers := []llm.Client{openaiClient, geminiClient, ollamaClient}
for _, provider := range providers {
    if err := checkProviderHealth(provider); err != nil {
        // Alert, log, or take corrective action
        log.Printf("Health check failed: %v", err)
    }
}
```

**Performance Benefits:**

- **Cached results** prevent API rate limit exhaustion
- **5-minute intervals** balance freshness with efficiency
- **Lightweight checks** minimize provider resource usage
- **Thread-safe** caching per client instance

## Retry with Exponential Backoff

Wrap any client with automatic retry functionality to handle transient errors like rate limiting (HTTP 429) and server errors (5xx).

### Basic Usage (Default Configuration)

```go
// Wrap any existing client with default retry settings
// Default: 3 retries, 1s base delay, 2x backoff, 60s max delay, jitter enabled
retryClient := llm.RetryChatCompletion(client)

resp, err := retryClient.ChatCompletion(ctx, req)
// Will automatically retry on rate limits and server errors
```

### Custom Configuration

```go
// Custom retry configuration for different scenarios
config := llm.RetryConfig{
    MaxRetries:    5,                // Allow up to 5 retries
    BaseDelay:     2 * time.Second,  // Start with 2 second delay
    MaxDelay:      30 * time.Second, // Cap delays at 30 seconds
    BackoffFactor: 2.0,              // Double delay each retry
    Jitter:        true,             // Add randomness to prevent thundering herd
}

retryClient := llm.RetryChatCompletion(client, config)
```

### Granular Error Control

Specify exactly which errors should trigger retries:

```go
// Only retry on rate limits (useful for integration tests)
testConfig := llm.RetryConfig{
    MaxRetries:         3,
    BaseDelay:          2 * time.Second,
    RetryOnStatusCodes: []int{429}, // Only HTTP 429 rate limit errors
}

// Retry on specific combinations
prodConfig := llm.RetryConfig{
    MaxRetries:         5,
    BaseDelay:          1 * time.Second,
    RetryOnStatusCodes: []int{429, 502, 503},          // Rate limits + specific server errors
    RetryOnErrorTypes:  []string{"rate_limit_error"},  // Plus rate limit error types
}

retryClient := llm.RetryChatCompletion(client, prodConfig)
```

### Usage with Context Timeout

```go
// Recommended: Use context timeout with retries
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
defer cancel()

retryClient := llm.RetryChatCompletion(client)
resp, err := retryClient.ChatCompletion(ctx, req)
// Will respect context cancellation during retry delays
```

**Automatically Retried Errors:**

- HTTP 429 (Rate limit exceeded)
- HTTP 5xx (Server errors - when using default behavior)
- Error types: "rate_limit_error"
- Custom error codes specified in `RetryableErrors`

**Non-Retried Errors:**

- HTTP 401/403 (Authentication/Authorization)
- HTTP 400 (Bad request/Invalid input)
- Network timeouts (respect context deadlines)
