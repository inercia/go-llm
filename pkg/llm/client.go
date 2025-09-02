// Client interfaces and core streaming functionality
package llm

import (
	"context"
)

// Client defines the core interface that all LLM clients must implement
type Client interface {
	// ChatCompletion performs a chat completion request
	ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error)

	// StreamChatCompletion performs a streaming chat completion request
	StreamChatCompletion(ctx context.Context, req ChatRequest) (<-chan StreamEvent, error)

	// GetModelInfo returns information about the model being used
	GetModelInfo() ModelInfo

	// Close cleans up any resources used by the client
	Close() error
}

// StreamingClient extends Client with tool stream injection capabilities
type StreamingClient interface {
	Client // Embed existing interface

	// StreamChatCompletionWithTools performs streaming with real-time tool injection
	StreamChatCompletionWithTools(ctx context.Context, req ChatRequest, toolStreams []<-chan StreamEvent) (<-chan StreamEvent, error)
}
