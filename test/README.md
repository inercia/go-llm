# Integration Tests

This directory contains comprehensive integration tests for the go-llm library, organized by functionality.

## Test Structure

### Core Test Files

- **`utils.go`** - Shared utilities and helper functions
  - `createTestClient()` - Creates client using environment configuration
  - Provider detection and capability checking
  - Fixture image loading utilities
  - Common test setup functions

### Functional Test Files

- **`chat_test.go`** - Basic chat functionality
  - Simple Q&A interactions
  - Conversation with history/memory
  - System message handling
  - Streaming responses
  - Error handling scenarios

- **`tools_test.go`** - Tool calling capabilities
  - Simple tool definitions and calls
  - Multiple tool selection
  - Tool conversation flows
  - Tool result handling
  - Streaming with tools
  - Tool error scenarios

- **`multimodal_test.go`** - Vision and file processing
  - Image description with fixture images
  - Multiple image comparison
  - File content analysis (JSON, text)
  - Streaming with images
  - Mixed content types
  - Error handling for invalid images

- **`streaming_test.go`** - Streaming-specific functionality
  - Basic streaming responses
  - Performance metrics (time to first token)
  - Long content streaming
  - Context cancellation during streaming
  - Event type validation

- **`factory_test.go`** - Factory and configuration
  - Environment-based configuration (`GetLLMFromEnv()`)
  - Multiple client creation
  - Custom configurations (timeouts, base URLs)
  - Provider capability validation
  - Error handling for invalid configs

- **`integration_test.go`** - Overall integration validation
  - End-to-end functionality test
  - Provider health checks
  - High-level integration validation

## Provider Selection

Tests automatically select the LLM provider based on environment variables:

1. **Priority 1**: Custom OpenAI-compatible endpoint (`OPENAI_BASE_URL` set)
2. **Priority 2**: OpenAI API (`OPENAI_API_KEY` set)
3. **Priority 3**: Gemini API (`GEMINI_API_KEY` set)
4. **Priority 4**: Ollama local (`http://localhost:11434`)

## Environment Variables

### OpenAI

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4o-mini"  # optional
export OPENAI_TIMEOUT="30"         # optional, seconds
```

### Custom OpenAI-Compatible

```bash
export OPENAI_BASE_URL="https://your-endpoint.com/v1"
export OPENAI_API_KEY="your-api-key"
export MODEL="your-model-name"     # optional
```

### Gemini

```bash
export GEMINI_API_KEY="your-api-key"
export GEMINI_MODEL="gemini-1.5-flash"  # optional
export GEMINI_TIMEOUT="30"              # optional, seconds
```

### Ollama

```bash
export OLLAMA_TIMEOUT="60"  # optional, seconds
# No API key needed - uses local server at http://localhost:11434
```

## Running Tests

### All Tests

```bash
go test ./test -v
```

### Specific Functionality

```bash
# Chat functionality
go test ./test -run TestChat -v

# Tool calling
go test ./test -run TestTools -v

# Multimodal capabilities
go test ./test -run TestMultiModal -v

# Streaming functionality
go test ./test -run TestStreaming -v

# Factory and configuration
go test ./test -run TestFactory -v

# Overall integration
go test ./test -run TestIntegrationOverall -v
```

### With Specific Provider

```bash
# Test with OpenAI
OPENAI_API_KEY=your-key go test ./test -v

# Test with Gemini
GEMINI_API_KEY=your-key go test ./test -v

# Test with Ollama (requires running server)
go test ./test -v
```

## Test Features

### Smart Provider Detection

- Automatically skips tests if no provider is available
- Capability checking (vision, tools, streaming support)
- Provider-specific test adaptations

### Comprehensive Coverage

- **Basic Chat**: Q&A, conversation memory, system messages
- **Advanced Features**: Tool calling, multimodal inputs, streaming
- **Error Handling**: Invalid inputs, timeouts, cancellation
- **Performance**: Response times, streaming metrics

### Real Provider Testing

- Uses actual LLM providers (not mocks)
- Tests end-to-end functionality
- Validates provider-specific behavior
- Fixture-based multimodal testing

### Robust Error Handling

- Graceful degradation when providers unavailable
- Proper error type validation
- Timeout and cancellation testing
- Invalid input handling

## Fixture Images

The `fixtures/` directory contains test images for multimodal testing:

- Various formats: JPEG, PNG
- Different sizes: small patterns to large photos
- Content types: cars, trees, patterns, text, geometric shapes

## Test Design Principles

1. **Environment-Driven**: Uses real providers via environment configuration
2. **Capability-Aware**: Adapts tests based on provider capabilities
3. **Comprehensive**: Covers all major functionality areas
4. **Robust**: Handles errors gracefully and provides clear diagnostics
5. **Performance-Conscious**: Includes timing and performance validation
6. **Well-Organized**: Split by functionality for maintainability
