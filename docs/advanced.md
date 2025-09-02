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
