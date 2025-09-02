// Client interfaces and core streaming functionality
package llm

import (
	"context"
	"time"
)

// DefaultHealthCheckInterval defines how often health checks should be refreshed
// to avoid excessive API calls to remote providers
const DefaultHealthCheckInterval = 5 * time.Minute

// ClientRemoteInfo represents information about a remote client
type ClientRemoteInfo struct {
	Name   string
	Status *ClientRemoteInfoStatus
}

// ClientRemoteInfoStatus represents the status of a remote client
type ClientRemoteInfoStatus struct {
	Healthy     *bool
	LastChecked *time.Time
}

// Client defines the core interface that all LLM clients must implement
type Client interface {
	// ChatCompletion performs a chat completion request
	ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error)

	// StreamChatCompletion performs a streaming chat completion request
	StreamChatCompletion(ctx context.Context, req ChatRequest) (<-chan StreamEvent, error)

	// GetRemoteInfo returns information about the client
	GetRemote() ClientRemoteInfo

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
