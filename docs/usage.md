# Usage Guide

This guide covers the basic usage patterns for the LLM client library. It assumes you have the package imported as `github.com/inercia/go-llm/pkg/llm`.

## Installation

Add the package to your Go module:

```bash
go get github.com/inercia/go-llm
```

## Automatic Configuration from Environment Variables

The library can automatically detect and configure clients based on environment variables, eliminating the need for manual provider selection. Use `llm.GetLLMFromEnv()` to get a configuration that will be automatically selected based on available credentials.

### Priority Order

The library checks for credentials in this order (first match wins):

1. **Custom OpenAI-compatible endpoint** (if `OPENAI_BASE_URL` is set)
2. **OpenAI API** (if `OPENAI_API_KEY` is set)
3. **Gemini API** (if `GEMINI_API_KEY` is set)
4. **DeepSeek API** (if `DEEPSEEK_API_KEY` is set)
5. **OpenRouter API** (if `OPENROUTER_API_KEY` is set)
6. **AWS Bedrock** (if AWS credentials are available)
7. **Ollama** (local fallback)

### Environment Variables

#### OpenAI / Custom OpenAI-compatible

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_MODEL="gpt-4o"                    # optional, defaults to gpt-4o-mini
export OPENAI_TIMEOUT="30"                      # optional, seconds
export OPENAI_BASE_URL="http://localhost:8080"  # for custom endpoints
```

#### Gemini

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export GEMINI_MODEL="gemini-1.5-pro"           # optional, defaults to gemini-1.5-flash
export GEMINI_TIMEOUT="30"                     # optional, seconds
```

#### DeepSeek

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export DEEPSEEK_MODEL="deepseek-coder"         # optional, defaults to deepseek-chat
export DEEPSEEK_TIMEOUT="30"                   # optional, seconds
```

#### OpenRouter

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
export OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"  # optional, defaults to free llama model
export OPENROUTER_TIMEOUT="30"                         # optional, seconds
```

#### AWS Bedrock

```bash
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export AWS_REGION="us-east-1"                          # optional, defaults to us-east-1
export AWS_BEDROCK_MODEL="anthropic.claude-3-sonnet-20240229-v1:0"  # optional
export BEDROCK_MODEL="anthropic.claude-3-haiku-20240307-v1:0"       # alternative
export AWS_BEDROCK_TIMEOUT="60"                        # optional, seconds
# Endpoint configuration (optional)
export AWS_BEDROCK_ENDPOINT="https://bedrock.custom.amazonaws.com"          # bedrock service endpoint
export AWS_BEDROCK_RUNTIME_ENDPOINT="https://bedrock-runtime.custom.amazonaws.com"  # runtime endpoint
export BEDROCK_ENDPOINT="https://bedrock-runtime.custom.amazonaws.com"      # alternative runtime endpoint
```

#### Ollama (Local Fallback)

```bash
export OLLAMA_TIMEOUT="60"                     # optional, seconds
# Ollama runs on http://localhost:11434 by default
```

### Usage Example

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/inercia/go-llm/pkg/factory"
    "github.com/inercia/go-llm/pkg/llm"
)

func main() {
    // Automatically detect provider from environment
    config := llm.GetLLMFromEnv()
    fmt.Printf("Using provider: %s with model: %s\n", config.Provider, config.Model)

    // Create client with auto-detected config
    client, err := factory.New().CreateClient(config)
    if err != nil {
        log.Fatal(err)
    }

    // Use the client normally
    resp, err := client.ChatCompletion(context.Background(), llm.ChatRequest{
        Messages: []llm.Message{
            llm.NewTextMessage(llm.RoleUser, "Hello!"),
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(resp.Choices[0].Message.GetText())
}
```

### Manual Configuration

You can still manually specify provider configurations if needed:

```go
client, err := factory.New().CreateClient(llm.ClientConfig{
    Provider: "bedrock",
    Model:    "anthropic.claude-3-sonnet-20240229-v1:0",
    Extra: map[string]string{
        "region": "us-west-2",
    },
})
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
