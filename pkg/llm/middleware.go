package llm

import (
	"context"
	"fmt"
	"sync"
)

// Middleware defines the interface for LLM middleware components
type Middleware interface {
	// Name returns the middleware name for identification
	Name() string

	// ProcessRequest processes the request before sending to LLM
	ProcessRequest(ctx context.Context, req *ChatRequest) (*ChatRequest, error)

	// ProcessResponse processes the response after receiving from LLM
	ProcessResponse(ctx context.Context, req *ChatRequest, resp *ChatResponse, err error) (*ChatResponse, error)

	// ProcessStreamEvent processes streaming events
	ProcessStreamEvent(ctx context.Context, req *ChatRequest, event StreamEvent) (StreamEvent, error)
}

// MiddlewareChain manages a chain of LLM middleware
type MiddlewareChain struct {
	mu          sync.RWMutex
	middlewares []Middleware
}

// NewMiddlewareChain creates a new middleware chain with default middleware
func NewMiddlewareChain(middlewares []Middleware) *MiddlewareChain {
	chain := &MiddlewareChain{}
	for _, middleware := range middlewares {
		chain.AddMiddleware(middleware)
	}
	return chain
}

// AddMiddleware adds a middleware to the chain
func (c *MiddlewareChain) AddMiddleware(middleware Middleware) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.middlewares = append(c.middlewares, middleware)
}

// RemoveMiddleware removes a middleware by name
func (c *MiddlewareChain) RemoveMiddleware(name string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	for i, middleware := range c.middlewares {
		if middleware.Name() == name {
			c.middlewares = append(c.middlewares[:i], c.middlewares[i+1:]...)
			return true
		}
	}
	return false
}

// ProcessRequest processes request through the middleware chain
func (c *MiddlewareChain) ProcessRequest(ctx context.Context, req *ChatRequest) (*ChatRequest, error) {
	c.mu.RLock()
	middlewares := make([]Middleware, len(c.middlewares))
	copy(middlewares, c.middlewares)
	c.mu.RUnlock()

	currentReq := req
	var err error

	for _, middleware := range middlewares {
		currentReq, err = middleware.ProcessRequest(ctx, currentReq)
		if err != nil {
			return nil, fmt.Errorf("middleware %s failed: %w", middleware.Name(), err)
		}
	}

	return currentReq, nil
}

// ProcessResponse processes response through the middleware chain (in reverse order)
func (c *MiddlewareChain) ProcessResponse(ctx context.Context, req *ChatRequest, resp *ChatResponse, err error) (*ChatResponse, error) {
	c.mu.RLock()
	middlewares := make([]Middleware, len(c.middlewares))
	copy(middlewares, c.middlewares)
	c.mu.RUnlock()

	currentResp := resp
	currentErr := err

	// Process in reverse order
	for i := len(middlewares) - 1; i >= 0; i-- {
		middleware := middlewares[i]
		processedResp, processErr := middleware.ProcessResponse(ctx, req, currentResp, currentErr)
		if processErr != nil {
			// Continue with other middleware even if one fails
			continue
		}
		currentResp = processedResp
	}

	return currentResp, currentErr
}

// ProcessStreamEvent processes stream events through the middleware chain
func (c *MiddlewareChain) ProcessStreamEvent(ctx context.Context, req *ChatRequest, event StreamEvent) (StreamEvent, error) {
	c.mu.RLock()
	middlewares := make([]Middleware, len(c.middlewares))
	copy(middlewares, c.middlewares)
	c.mu.RUnlock()

	currentEvent := event
	var err error

	for _, middleware := range middlewares {
		currentEvent, err = middleware.ProcessStreamEvent(ctx, req, currentEvent)
		if err != nil {
			// Continue with other middleware even if one fails
			continue
		}
	}

	return currentEvent, nil
}

// GetMiddlewareNames returns the names of all middleware in the chain
func (c *MiddlewareChain) GetMiddlewareNames() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	names := make([]string, len(c.middlewares))
	for i, middleware := range c.middlewares {
		names[i] = middleware.Name()
	}
	return names
}
