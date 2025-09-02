package llm

import (
	"context"
	"fmt"
)

// EnhancedClient wraps an LLM client with middleware chain
type EnhancedClient struct {
	client Client
	chain  *MiddlewareChain
}

// NewEnhancedClient creates a new enhanced LLM client with middleware
func NewEnhancedClient(client Client, chain []Middleware) *EnhancedClient {
	return &EnhancedClient{
		client: client,
		chain:  NewMiddlewareChain(chain),
	}
}

// ChatCompletion implements Client interface with middleware processing
func (e *EnhancedClient) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	// Process request through middleware chain
	processedReq, err := e.chain.ProcessRequest(ctx, &req)
	if err != nil {
		return nil, fmt.Errorf("middleware request processing failed: %w", err)
	}

	// Execute the actual LLM call
	resp, err := e.client.ChatCompletion(ctx, *processedReq)

	// Process response through middleware chain
	processedResp, _ := e.chain.ProcessResponse(ctx, processedReq, resp, err)

	return processedResp, err
}

// StreamChatCompletion implements Client interface with middleware processing for streaming
func (e *EnhancedClient) StreamChatCompletion(ctx context.Context, req ChatRequest) (<-chan StreamEvent, error) {
	// Process request through middleware chain
	processedReq, err := e.chain.ProcessRequest(ctx, &req)
	if err != nil {
		return nil, fmt.Errorf("middleware request processing failed: %w", err)
	}

	// Execute the actual LLM streaming call
	eventChan, err := e.client.StreamChatCompletion(ctx, *processedReq)
	if err != nil {
		// Process error through middleware
		_, _ = e.chain.ProcessResponse(ctx, processedReq, nil, err)
		return nil, err
	}

	// Create a new channel for processed events
	processedChan := make(chan StreamEvent)

	go func() {
		defer close(processedChan)

		for event := range eventChan {
			// Process each stream event through middleware
			processedEvent, processErr := e.chain.ProcessStreamEvent(ctx, processedReq, event)
			if processErr != nil {
				// Send original event if processing fails
				processedEvent = event
			}

			select {
			case processedChan <- processedEvent:
			case <-ctx.Done():
				return
			}
		}

		// Process final response through middleware (for completion tracking)
		_, _ = e.chain.ProcessResponse(ctx, processedReq, nil, nil)
	}()

	return processedChan, nil
}

// GetRemote implements Client interface
func (e *EnhancedClient) GetRemote() ClientRemoteInfo {
	return e.client.GetRemote()
}

// GetModelInfo implements Client interface
func (e *EnhancedClient) GetModelInfo() ModelInfo {
	return e.client.GetModelInfo()
}

// Close implements Client interface
func (e *EnhancedClient) Close() error {
	return e.client.Close()
}

// AddMiddleware adds a middleware to the client's chain
func (e *EnhancedClient) AddMiddleware(middleware Middleware) {
	e.chain.AddMiddleware(middleware)
}

// RemoveMiddleware removes a middleware from the client's chain
func (e *EnhancedClient) RemoveMiddleware(name string) bool {
	return e.chain.RemoveMiddleware(name)
}

// GetMiddlewareNames returns the names of all middleware in the client's chain
func (e *EnhancedClient) GetMiddlewareNames() []string {
	return e.chain.GetMiddlewareNames()
}

// ClientWithMiddleware wraps an existing LLM client with the enhanced middleware system
// This is the main entry point for adding middleware to LLM clients
func ClientWithMiddleware(client Client, chain []Middleware) Client {
	// If the client is already an EnhancedClient, add middleware to existing chain
	if enhancedClient, ok := client.(*EnhancedClient); ok {
		for _, middleware := range chain {
			enhancedClient.AddMiddleware(middleware)
		}
		return enhancedClient
	}

	return NewEnhancedClient(client, chain)
}
