# AWS Bedrock Provider

The AWS Bedrock provider integrates with Amazon Bedrock using the official AWS SDK for Go v2. It provides unified access to multiple foundation models including Claude, Titan, and Llama through a single interface.

## Features

- **Multiple Model Support**: Works with Claude (Anthropic), Titan (Amazon), and Llama (Meta) models
- **Chat Completions**: Full support for multi-turn conversations with automatic model-specific format conversion
- **Streaming**: Real-time token-by-token responses for all supported models
- **Multi-modal Support**: Image inputs for Claude 3 models that support vision
- **Error Standardization**: Maps AWS Bedrock errors to the library's `llm.Error` structure
- **Health Checks**: Monitors AWS connectivity and permissions
- **Regional Support**: Configurable AWS region selection
- **Authentication**: Uses AWS credential chain (IAM roles, profiles, environment variables)

## Setup

### 1. AWS Configuration

Set up AWS credentials using one of these methods:

**Environment Variables:**

```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION=us-east-1  # Optional, defaults to us-east-1
```

**AWS Profile:**

```bash
aws configure --profile your-profile
export AWS_PROFILE=your-profile
```

**IAM Role:** (when running on EC2, ECS, Lambda, etc.)

### 2. Bedrock Model Access

Request access to foundation models in the AWS Bedrock console:

1. Navigate to AWS Bedrock in your preferred region
2. Go to "Model access" in the left sidebar
3. Request access to the models you want to use (e.g., Claude, Titan)
4. Wait for approval (usually immediate for most models)

### 3. Create Client

```go
import (
    "github.com/inercia/go-llm/pkg/factory"
    "github.com/inercia/go-llm/pkg/llm"
)

client, err := factory.CreateClient(llm.ClientConfig{
    Provider: "bedrock",
    Model:    "anthropic.claude-3-sonnet-20240229-v1:0",
    Extra: map[string]string{
        "region": "us-east-1", // Optional, defaults to us-east-1
        // Optional custom endpoints
        "bedrock_endpoint": "https://bedrock.us-west-2.amazonaws.com",
        "bedrock_runtime_endpoint": "https://bedrock-runtime.us-west-2.amazonaws.com",
    },
})

// Alternative using BaseURL (applies to bedrock-runtime endpoint)
client, err := factory.CreateClient(llm.ClientConfig{
    Provider: "bedrock",
    Model:    "anthropic.claude-3-sonnet-20240229-v1:0",
    BaseURL:  "https://bedrock-runtime.us-west-2.amazonaws.com",
    Extra: map[string]string{
        "region": "us-west-2",
    },
})
```

## Supported Models

### Claude Models (Anthropic)

**Claude 3.5 Sonnet** (Recommended)

- Model ID: `anthropic.claude-3-5-sonnet-20241022-v2:0`
- Features: Text + Vision, 200k context, Function calling
- Best for: Complex reasoning, analysis, content creation

**Claude 3 Sonnet**

- Model ID: `anthropic.claude-3-sonnet-20240229-v1:0`
- Features: Text + Vision, 200k context, Function calling
- Best for: Balanced performance and cost

**Claude 3 Haiku**

- Model ID: `anthropic.claude-3-haiku-20240307-v1:0`
- Features: Text + Vision, 200k context, Fast response
- Best for: Quick tasks, high throughput

**Claude v2.1**

- Model ID: `anthropic.claude-v2:1`
- Features: Text only, 100k context
- Best for: Legacy applications

### Titan Models (Amazon)

**Titan Text G1 - Express**

- Model ID: `amazon.titan-text-express-v1`
- Features: Text only, 8k context
- Best for: Fast text generation, summarization

**Titan Text G1 - Lite**

- Model ID: `amazon.titan-text-lite-v1`
- Features: Text only, 4k context
- Best for: Simple tasks, cost optimization

### Llama Models (Meta)

**Llama 2 70B Chat**

- Model ID: `meta.llama2-70b-chat-v1`
- Features: Text only, 4k context
- Best for: Open-source alternative, coding tasks

**Llama 2 13B Chat**

- Model ID: `meta.llama2-13b-chat-v1`
- Features: Text only, 2k context
- Best for: Lightweight applications

## Usage Examples

### Basic Chat Completion

```go
ctx := context.Background()

response, err := client.ChatCompletion(ctx, llm.ChatRequest{
    Model: "anthropic.claude-3-sonnet-20240229-v1:0",
    Messages: []llm.Message{
        llm.NewTextMessage(llm.RoleUser, "Hello! How are you?"),
    },
    MaxTokens:   aws.Int(1000),
    Temperature: aws.Float32(0.7),
})

if err != nil {
    log.Fatal(err)
}

fmt.Println(response.Choices[0].Message.GetText())
```

### Streaming Chat

```go
stream, err := client.StreamChatCompletion(ctx, llm.ChatRequest{
    Model: "anthropic.claude-3-sonnet-20240229-v1:0",
    Messages: []llm.Message{
        llm.NewTextMessage(llm.RoleUser, "Write a short story about space exploration."),
    },
    MaxTokens: aws.Int(2000),
})

if err != nil {
    log.Fatal(err)
}

for event := range stream {
    if event.IsError() {
        log.Printf("Error: %v", event.Error)
        break
    }

    if event.IsDelta() {
        fmt.Print(event.Delta.GetText())
    }

    if event.IsDone() {
        fmt.Println("\n[Completed]")
        break
    }
}
```

### Multi-modal with Claude 3

```go
// Load image data
imageData, _ := os.ReadFile("image.jpg")

response, err := client.ChatCompletion(ctx, llm.ChatRequest{
    Model: "anthropic.claude-3-sonnet-20240229-v1:0",
    Messages: []llm.Message{
        {
            Role: llm.RoleUser,
            Content: []llm.MessageContent{
                llm.NewTextContent("What do you see in this image?"),
                llm.NewImageContent(imageData, "image/jpeg"),
            },
        },
    },
    MaxTokens: aws.Int(1000),
})
```

## Configuration Options

| Option                              | Type      | Description                          | Default       |
| ----------------------------------- | --------- | ------------------------------------ | ------------- |
| `Model`                             | string    | Bedrock model ID                     | Required      |
| `BaseURL`                           | string    | Bedrock Runtime endpoint URL         | AWS default   |
| `Extra["region"]`                   | string    | AWS region                           | `us-east-1`   |
| `Extra["bedrock_endpoint"]`         | string    | Bedrock service endpoint URL         | AWS default   |
| `Extra["bedrock_runtime_endpoint"]` | string    | Bedrock Runtime service endpoint URL | AWS default   |
| `Extra["base_url"]`                 | string    | Alternative runtime endpoint         | AWS default   |
| `MaxTokens`                         | \*int     | Maximum tokens to generate           | Model default |
| `Temperature`                       | \*float32 | Response randomness (0.0-1.0)        | Model default |
| `TopP`                              | \*float32 | Nucleus sampling parameter           | Model default |

## Known Issues and Limitations

### Authentication Issues

- **Access Denied**: Ensure your AWS credentials have `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` permissions
- **Model Access**: Request access to specific models in the Bedrock console
- **Region Mismatch**: Ensure you're using the correct region where you have model access

### Model-Specific Limitations

**Claude Models:**

- Function calling only supported on Claude 3+ models
- Vision support only on Claude 3+ models
- Legacy prompt format used for Claude v2

**Titan Models:**

- No function calling support
- Limited context lengths compared to Claude
- Simple prompt-response format

**Llama Models:**

- No function calling or vision support
- Specific prompt format required for chat
- Limited context length

### Performance Considerations

- **Cold Starts**: First request may have higher latency (1-3 seconds)
- **Streaming Latency**: Streaming responses typically start within 500-1000ms
- **Rate Limits**: Vary by model and region; implement exponential backoff
- **Token Limits**: Each model has different input/output token limits

### Error Handling

```go
if err != nil {
    if bedrockErr, ok := err.(*llm.Error); ok {
        switch bedrockErr.Code {
        case "authentication_error":
            log.Println("Check your AWS credentials")
        case "rate_limit_error":
            log.Println("Rate limited, retry with backoff")
        case "model_not_found":
            log.Println("Model not available or access not granted")
        default:
            log.Printf("Bedrock error: %v", bedrockErr.Message)
        }
    }
}
```

## Testing

For integration testing:

1. Set up AWS credentials with Bedrock access
2. Ensure model access is granted in your test region
3. Use environment variables or AWS profiles for CI/CD

```go
func TestBedrockIntegration(t *testing.T) {
    // Skip if no AWS credentials
    if os.Getenv("AWS_ACCESS_KEY_ID") == "" {
        t.Skip("No AWS credentials available")
    }

    client, err := factory.CreateClient(llm.ClientConfig{
        Provider: "bedrock",
        Model:    "anthropic.claude-3-haiku-20240307-v1:0", // Fast model for testing
    })

    require.NoError(t, err)

    // Test basic completion
    resp, err := client.ChatCompletion(ctx, llm.ChatRequest{
        Messages: []llm.Message{
            llm.NewTextMessage(llm.RoleUser, "Hello"),
        },
        MaxTokens: aws.Int(10),
    })

    require.NoError(t, err)
    require.NotEmpty(t, resp.Choices[0].Message.GetText())
}
```

## Pricing

AWS Bedrock pricing varies by model and region. Check the [AWS Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) for current rates.

Generally:

- **Claude models**: Higher cost, better quality
- **Titan models**: Lower cost, good for simple tasks
- **Llama models**: Moderate cost, open-source alternative

See the [main usage guide](../usage.md) for general examples and patterns.
