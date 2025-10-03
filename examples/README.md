# Go-LLM Examples

This directory contains practical examples demonstrating key features of the go-llm library. Each example is self-contained and shows different aspects of working with Large Language Models in Go.

## Available Examples

### [🌊 Streaming Example](streaming/)

**Real-time chat completions with streaming responses**

Demonstrates how to:

- Set up streaming chat completions
- Handle real-time text generation
- Process stream events (`IsDelta`, `IsDone`, `IsError`)
- Compare streaming vs non-streaming modes
- Accumulate streamed content while showing progress

**Use case**: Perfect for chatbots, interactive applications, or any scenario where you want to show AI responses as they're being generated.

### [📊 Structured Output Example](structured/)

**JSON Schema validation and typed responses**

Demonstrates how to:

- Generate JSON schemas from Go structs
- Get reliable, typed responses from LLMs
- Validate responses against schemas
- Handle complex data structures with nested objects and arrays
- Use struct tags for field constraints and descriptions

**Use cases**:

- Data extraction from unstructured text
- Form filling and structured data generation
- API response parsing
- Content analysis with consistent output formats

## Running the Examples

Each example directory contains:

- `main.go` - The example code
- `README.md` - Detailed explanation and usage instructions

### Prerequisites

1. **Go 1.21+** installed on your system
2. **API Key** for your chosen provider (OpenAI, Gemini, etc.)
3. **Dependencies** - run `go mod tidy` in the project root

### Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd go-llm

# Install dependencies
go mod tidy

# Set your API key for any provider (the examples auto-detect)
export OPENAI_API_KEY="your-api-key-here"  # or GEMINI_API_KEY, DEEPSEEK_API_KEY, etc.

# Run streaming example
cd examples/streaming
go run main.go

# Run structured output example
cd ../structured
go run main.go
```

## Provider Flexibility

Both examples work with **any supported provider** using automatic detection from environment variables:

```go
// Automatically detects provider based on available API keys
config := llm.GetLLMFromEnv()
client, err := factory.CreateClient(config)
```

Supported environment variables (in priority order):

- `OPENAI_API_KEY` - OpenAI API
- `GEMINI_API_KEY` - Google Gemini API
- `DEEPSEEK_API_KEY` - DeepSeek API
- `OPENROUTER_API_KEY` - OpenRouter API
- AWS credentials - Bedrock API
- Falls back to local Ollama if no keys are found

See the [main documentation](../docs/README.md) for provider-specific setup instructions.

## Next Steps

- Read the [Architecture Guide](../docs/architecture.md) to understand the library design
- Check [Provider Documentation](../docs/providers/) for setup instructions
- Explore the [Advanced Usage](../docs/advanced.md) guide for more complex scenarios
- Review the [API Reference](../pkg/) for complete interface documentation

## Contributing Examples

Have an idea for a useful example? We welcome contributions! Please:

1. Create a new directory under `examples/`
2. Include a complete `main.go` with clear comments
3. Add a detailed `README.md` explaining the example
4. Update this index file to include your example
