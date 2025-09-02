// Package deepseek provides an LLM client for DeepSeek models.
//
// This provider implements the llm.Client interface for DeepSeek's API,
// supporting both streaming and non-streaming chat completions with
// comprehensive tool calling capabilities.
//
// Key features:
//   - Text-based chat completions with streaming support
//   - Tool calling and function execution support
//   - Comprehensive error handling and validation
//   - Configurable timeouts and model selection
//   - Image and file content handling (converted to text descriptions)
//
// The client automatically registers itself with the LLM provider registry
// during package initialization, making it available for use with the
// factory pattern.
//
// Usage:
//
//	config := llm.ClientConfig{
//	    Provider: "deepseek",
//	    APIKey:   "your-api-key",
//	    Model:    "deepseek-chat",
//	}
//	client, err := llm.CreateClient(config)
package deepseek
