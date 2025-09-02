// Package gemini provides an LLM client for Google Gemini models.
//
// This provider implements the llm.Client interface for Google's Gemini API
// using the official Google Generative AI library. It supports both streaming
// and non-streaming chat completions with text and image inputs.
//
// Key features:
//   - Text and multimodal (text + image) content support
//   - Streaming chat completions
//   - Automatic error conversion to standardized format
//   - Response format instructions via prompt engineering
//   - Temperature and token limit controls
//
// The client automatically registers itself with the LLM provider registry
// during package initialization, making it available for use with the
// factory pattern.
//
// Usage:
//
//	config := llm.ClientConfig{
//	    Provider: "gemini",
//	    APIKey:   "your-api-key",
//	    Model:    "gemini-1.5-flash",
//	}
//	client, err := llm.CreateClient(config)
package gemini
