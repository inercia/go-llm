package llm

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

// Mock implementations for testing

// testMockClient is a simple mock client for testing
type testMockClient struct {
	model         string
	provider      string
	responses     []*ChatResponse
	streamEvents  []StreamEvent
	errorToReturn error
	callLog       []ChatRequest
}

func NewMockClient(model, provider string) *testMockClient {
	return &testMockClient{
		model:    model,
		provider: provider,
	}
}

func (c *testMockClient) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	c.callLog = append(c.callLog, req)

	if c.errorToReturn != nil {
		return nil, c.errorToReturn
	}

	if len(c.responses) > 0 {
		resp := c.responses[0]
		c.responses = c.responses[1:]
		return resp, nil
	}

	return &ChatResponse{
		ID:    "test-response",
		Model: c.model,
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    RoleAssistant,
					Content: []MessageContent{NewTextContent("Test response")},
				},
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}, nil
}

func (c *testMockClient) StreamChatCompletion(ctx context.Context, req ChatRequest) (<-chan StreamEvent, error) {
	c.callLog = append(c.callLog, req)

	if c.errorToReturn != nil {
		return nil, c.errorToReturn
	}

	ch := make(chan StreamEvent, len(c.streamEvents)+1)
	go func() {
		defer close(ch)
		if len(c.streamEvents) > 0 {
			for _, event := range c.streamEvents {
				ch <- event
			}
		} else {
			// Default stream response
			ch <- NewDeltaEvent(0, &MessageDelta{
				Content: []MessageContent{NewTextContent("Test")},
			})
			ch <- NewDoneEvent(0, "stop")
		}
	}()
	return ch, nil
}

func (c *testMockClient) GetModelInfo() ModelInfo {
	return ModelInfo{
		Name:              c.model,
		Provider:          c.provider,
		MaxTokens:         4096,
		SupportsTools:     true,
		SupportsVision:    false,
		SupportsFiles:     false,
		SupportsStreaming: true,
	}
}

func (c *testMockClient) GetRemote() ClientRemoteInfo {
	healthy := true // Test mock is always healthy
	now := time.Now()
	return ClientRemoteInfo{
		Name: c.provider,
		Status: &ClientRemoteInfoStatus{
			Healthy:     &healthy,
			LastChecked: &now,
		},
	}
}

func (c *testMockClient) Close() error {
	return nil
}

// Additional mock methods for testing
func (c *testMockClient) AddResponse(resp ChatResponse) {
	c.responses = append(c.responses, &resp)
}

func (c *testMockClient) WithSimpleResponse(text string) *testMockClient {
	c.AddResponse(ChatResponse{
		ID:    "simple-response",
		Model: c.model,
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    RoleAssistant,
					Content: []MessageContent{NewTextContent(text)},
				},
				FinishReason: "stop",
			},
		},
	})
	return c
}

func (c *testMockClient) WithError(code, message, errorType string) *testMockClient {
	c.errorToReturn = &Error{
		Code:    code,
		Message: message,
		Type:    errorType,
	}
	return c
}

func (c *testMockClient) WithStreamResponse(events []StreamEvent) *testMockClient {
	c.streamEvents = events
	return c
}

func (c *testMockClient) GetCallLog() []ChatRequest {
	return c.callLog
}

// mockMiddleware is a basic mock implementation of the Middleware interface
type mockMiddleware struct {
	name      string
	reqMods   func(*ChatRequest) (*ChatRequest, error)
	respMods  func(*ChatRequest, *ChatResponse, error) (*ChatResponse, error)
	eventMods func(StreamEvent) (StreamEvent, error)
}

func (m *mockMiddleware) Name() string {
	return m.name
}

func (m *mockMiddleware) ProcessRequest(ctx context.Context, req *ChatRequest) (*ChatRequest, error) {
	if m.reqMods != nil {
		return m.reqMods(req)
	}
	return req, nil
}

func (m *mockMiddleware) ProcessResponse(ctx context.Context, req *ChatRequest, resp *ChatResponse, err error) (*ChatResponse, error) {
	if m.respMods != nil {
		return m.respMods(req, resp, err)
	}
	return resp, err
}

func (m *mockMiddleware) ProcessStreamEvent(ctx context.Context, req *ChatRequest, event StreamEvent) (StreamEvent, error) {
	if m.eventMods != nil {
		return m.eventMods(event)
	}
	return event, nil
}

// newTestMiddleware creates a simple test middleware with the given name
func newTestMiddleware(name string) *mockMiddleware {
	return &mockMiddleware{name: name}
}

// newErrorMiddleware creates a middleware that returns errors during processing
func newErrorMiddleware(name string) *mockMiddleware {
	return &mockMiddleware{
		name: name,
		reqMods: func(*ChatRequest) (*ChatRequest, error) {
			return nil, errors.New("request processing error")
		},
		respMods: func(*ChatRequest, *ChatResponse, error) (*ChatResponse, error) {
			return nil, errors.New("response processing error")
		},
		eventMods: func(StreamEvent) (StreamEvent, error) {
			return StreamEvent{}, errors.New("event processing error")
		},
	}
}

// newModifyingMiddleware creates a middleware that modifies the request/response
func newModifyingMiddleware(name string, modelModifier string) *mockMiddleware {
	return &mockMiddleware{
		name: name,
		reqMods: func(req *ChatRequest) (*ChatRequest, error) {
			modifiedReq := *req
			modifiedReq.Model = req.Model + modelModifier
			return &modifiedReq, nil
		},
		respMods: func(req *ChatRequest, resp *ChatResponse, err error) (*ChatResponse, error) {
			if resp != nil {
				modifiedResp := *resp
				modifiedResp.Model = resp.Model + modelModifier
				return &modifiedResp, err
			}
			return resp, err
		},
		eventMods: func(event StreamEvent) (StreamEvent, error) {
			modifiedEvent := event
			modifiedEvent.Type = event.Type + modelModifier
			return modifiedEvent, nil
		},
	}
}

func TestNewMiddlewareChain(t *testing.T) {
	t.Run("creates empty chain when no middleware provided", func(t *testing.T) {
		chain := NewMiddlewareChain(nil)
		if chain == nil {
			t.Fatal("expected non-nil chain")
		}
		names := chain.GetMiddlewareNames()
		if len(names) != 0 {
			t.Errorf("expected empty chain, got %d middleware", len(names))
		}
	})

	t.Run("creates chain with provided middleware", func(t *testing.T) {
		m1 := newTestMiddleware("middleware1")
		m2 := newTestMiddleware("middleware2")

		chain := NewMiddlewareChain([]Middleware{m1, m2})
		names := chain.GetMiddlewareNames()

		expected := []string{"middleware1", "middleware2"}
		if !equalStringSlice(names, expected) {
			t.Errorf("expected %v, got %v", expected, names)
		}
	})
}

func TestMiddlewareChain_AddMiddleware(t *testing.T) {
	chain := NewMiddlewareChain(nil)

	m1 := newTestMiddleware("middleware1")
	chain.AddMiddleware(m1)

	names := chain.GetMiddlewareNames()
	if len(names) != 1 || names[0] != "middleware1" {
		t.Errorf("expected [middleware1], got %v", names)
	}

	// Add another middleware
	m2 := newTestMiddleware("middleware2")
	chain.AddMiddleware(m2)

	names = chain.GetMiddlewareNames()
	expected := []string{"middleware1", "middleware2"}
	if !equalStringSlice(names, expected) {
		t.Errorf("expected %v, got %v", expected, names)
	}
}

func TestMiddlewareChain_RemoveMiddleware(t *testing.T) {
	m1 := newTestMiddleware("middleware1")
	m2 := newTestMiddleware("middleware2")
	m3 := newTestMiddleware("middleware3")

	chain := NewMiddlewareChain([]Middleware{m1, m2, m3})

	t.Run("removes existing middleware", func(t *testing.T) {
		removed := chain.RemoveMiddleware("middleware2")
		if !removed {
			t.Error("expected middleware to be removed")
		}

		names := chain.GetMiddlewareNames()
		expected := []string{"middleware1", "middleware3"}
		if !equalStringSlice(names, expected) {
			t.Errorf("expected %v, got %v", expected, names)
		}
	})

	t.Run("returns false for non-existent middleware", func(t *testing.T) {
		removed := chain.RemoveMiddleware("nonexistent")
		if removed {
			t.Error("expected false for non-existent middleware")
		}
	})

	t.Run("removes first middleware", func(t *testing.T) {
		chain := NewMiddlewareChain([]Middleware{m1, m2})
		removed := chain.RemoveMiddleware("middleware1")
		if !removed {
			t.Error("expected middleware to be removed")
		}

		names := chain.GetMiddlewareNames()
		if len(names) != 1 || names[0] != "middleware2" {
			t.Errorf("expected [middleware2], got %v", names)
		}
	})
}

func TestMiddlewareChain_ProcessRequest(t *testing.T) {
	ctx := context.Background()

	t.Run("processes through empty chain", func(t *testing.T) {
		chain := NewMiddlewareChain(nil)
		req := &ChatRequest{Model: "test-model"}

		result, err := chain.ProcessRequest(ctx, req)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if result != req {
			t.Error("expected same request object for empty chain")
		}
	})

	t.Run("processes through single middleware", func(t *testing.T) {
		m1 := newModifyingMiddleware("middleware1", "-modified")
		chain := NewMiddlewareChain([]Middleware{m1})
		req := &ChatRequest{Model: "test-model"}

		result, err := chain.ProcessRequest(ctx, req)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if result.Model != "test-model-modified" {
			t.Errorf("expected modified model, got %s", result.Model)
		}
	})

	t.Run("processes through multiple middleware in order", func(t *testing.T) {
		m1 := newModifyingMiddleware("middleware1", "-m1")
		m2 := newModifyingMiddleware("middleware2", "-m2")
		chain := NewMiddlewareChain([]Middleware{m1, m2})
		req := &ChatRequest{Model: "test"}

		result, err := chain.ProcessRequest(ctx, req)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if result.Model != "test-m1-m2" {
			t.Errorf("expected test-m1-m2, got %s", result.Model)
		}
	})

	t.Run("handles middleware error", func(t *testing.T) {
		m1 := newTestMiddleware("middleware1")
		m2 := newErrorMiddleware("error-middleware")
		chain := NewMiddlewareChain([]Middleware{m1, m2})
		req := &ChatRequest{Model: "test-model"}

		result, err := chain.ProcessRequest(ctx, req)
		if err == nil {
			t.Error("expected error from middleware")
		}
		if result != nil {
			t.Error("expected nil result on error")
		}
		if !errors.Is(err, errors.New("request processing error")) && !containsString(err.Error(), "error-middleware") {
			t.Errorf("expected error to mention middleware name, got: %v", err)
		}
	})
}

func TestMiddlewareChain_ProcessResponse(t *testing.T) {
	ctx := context.Background()

	t.Run("processes through empty chain", func(t *testing.T) {
		chain := NewMiddlewareChain(nil)
		req := &ChatRequest{Model: "test-model"}
		resp := &ChatResponse{Model: "test-model"}

		result, err := chain.ProcessResponse(ctx, req, resp, nil)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if result != resp {
			t.Error("expected same response object for empty chain")
		}
	})

	t.Run("processes through middleware in reverse order", func(t *testing.T) {
		m1 := newModifyingMiddleware("middleware1", "-m1")
		m2 := newModifyingMiddleware("middleware2", "-m2")
		chain := NewMiddlewareChain([]Middleware{m1, m2})
		req := &ChatRequest{Model: "test"}
		resp := &ChatResponse{Model: "test"}

		result, err := chain.ProcessResponse(ctx, req, resp, nil)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		// Should process in reverse order: m2 first, then m1
		if result.Model != "test-m2-m1" {
			t.Errorf("expected test-m2-m1, got %s", result.Model)
		}
	})

	t.Run("continues processing even if middleware fails", func(t *testing.T) {
		m1 := newModifyingMiddleware("middleware1", "-m1")
		m2 := newErrorMiddleware("error-middleware")
		m3 := newModifyingMiddleware("middleware3", "-m3")
		chain := NewMiddlewareChain([]Middleware{m1, m2, m3})
		req := &ChatRequest{Model: "test"}
		resp := &ChatResponse{Model: "test"}

		result, err := chain.ProcessResponse(ctx, req, resp, nil)
		// Should still get result from successful middleware
		if result == nil {
			t.Error("expected result even with middleware error")
		}
		// Should process m3, fail on m2, then process m1
		if result != nil && result.Model != "test-m3-m1" {
			t.Errorf("expected test-m3-m1, got %s", result.Model)
		}
		if err != nil {
			t.Errorf("unexpected error return: %v", err)
		}
	})

	t.Run("preserves input error", func(t *testing.T) {
		chain := NewMiddlewareChain([]Middleware{newTestMiddleware("test")})
		req := &ChatRequest{Model: "test"}
		inputErr := errors.New("input error")

		_, err := chain.ProcessResponse(ctx, req, nil, inputErr)
		if err != inputErr {
			t.Error("expected input error to be preserved")
		}
	})
}

func TestMiddlewareChain_ProcessStreamEvent(t *testing.T) {
	ctx := context.Background()

	t.Run("processes through empty chain", func(t *testing.T) {
		chain := NewMiddlewareChain(nil)
		req := &ChatRequest{Model: "test-model"}
		event := StreamEvent{Type: "delta"}

		result, err := chain.ProcessStreamEvent(ctx, req, event)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if result.Type != event.Type {
			t.Error("expected same event for empty chain")
		}
	})

	t.Run("processes through middleware in order", func(t *testing.T) {
		m1 := newModifyingMiddleware("middleware1", "-m1")
		m2 := newModifyingMiddleware("middleware2", "-m2")
		chain := NewMiddlewareChain([]Middleware{m1, m2})
		req := &ChatRequest{Model: "test"}
		event := StreamEvent{Type: "delta"}

		result, err := chain.ProcessStreamEvent(ctx, req, event)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if result.Type != "delta-m1-m2" {
			t.Errorf("expected delta-m1-m2, got %s", result.Type)
		}
	})

	t.Run("continues processing even if middleware fails", func(t *testing.T) {
		m1 := newModifyingMiddleware("middleware1", "-m1")
		m2 := newErrorMiddleware("error-middleware")
		m3 := newModifyingMiddleware("middleware3", "-m3")
		chain := NewMiddlewareChain([]Middleware{m1, m2, m3})
		req := &ChatRequest{Model: "test"}
		event := StreamEvent{Type: "delta"}

		result, err := chain.ProcessStreamEvent(ctx, req, event)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		// Should process m1, fail on m2 (which resets event to empty), then m3 processes empty event
		if result.Type != "-m3" {
			t.Errorf("expected -m3, got %s", result.Type)
		}
	})
}

func TestMiddlewareChain_GetMiddlewareNames(t *testing.T) {
	t.Run("returns empty slice for empty chain", func(t *testing.T) {
		chain := NewMiddlewareChain(nil)
		names := chain.GetMiddlewareNames()
		if len(names) != 0 {
			t.Errorf("expected empty slice, got %v", names)
		}
	})

	t.Run("returns middleware names in order", func(t *testing.T) {
		m1 := newTestMiddleware("first")
		m2 := newTestMiddleware("second")
		m3 := newTestMiddleware("third")
		chain := NewMiddlewareChain([]Middleware{m1, m2, m3})

		names := chain.GetMiddlewareNames()
		expected := []string{"first", "second", "third"}
		if !equalStringSlice(names, expected) {
			t.Errorf("expected %v, got %v", expected, names)
		}
	})
}

func TestMiddlewareChain_Concurrency(t *testing.T) {
	// Test concurrent access to middleware chain
	chain := NewMiddlewareChain(nil)
	ctx := context.Background()

	var wg sync.WaitGroup
	numGoroutines := 100

	// Test concurrent add/remove operations
	t.Run("concurrent add and remove", func(t *testing.T) {
		for i := 0; i < numGoroutines; i++ {
			wg.Add(2)

			go func(id int) {
				defer wg.Done()
				m := newTestMiddleware(fmt.Sprintf("middleware-%d", id))
				chain.AddMiddleware(m)
			}(i)

			go func(id int) {
				defer wg.Done()
				chain.RemoveMiddleware(fmt.Sprintf("middleware-%d", id))
			}(i)
		}

		wg.Wait()

		// Chain should still be functional
		names := chain.GetMiddlewareNames()
		if names == nil {
			t.Error("expected names slice to not be nil")
		}
	})

	// Test concurrent processing
	t.Run("concurrent processing", func(t *testing.T) {
		// Reset chain with some middleware
		chain = NewMiddlewareChain([]Middleware{
			newTestMiddleware("test1"),
			newTestMiddleware("test2"),
		})

		for i := 0; i < numGoroutines; i++ {
			wg.Add(3)

			go func() {
				defer wg.Done()
				req := &ChatRequest{Model: "test"}
				_, _ = chain.ProcessRequest(ctx, req)
			}()

			go func() {
				defer wg.Done()
				req := &ChatRequest{Model: "test"}
				resp := &ChatResponse{Model: "test"}
				_, _ = chain.ProcessResponse(ctx, req, resp, nil)
			}()

			go func() {
				defer wg.Done()
				req := &ChatRequest{Model: "test"}
				event := StreamEvent{Type: "delta"}
				_, _ = chain.ProcessStreamEvent(ctx, req, event)
			}()
		}

		wg.Wait()
	})
}

// TestClientWithMiddleware tests the ClientWithMiddleware wrapper function
func TestClientWithMiddleware(t *testing.T) {
	t.Run("wraps regular client with middleware", func(t *testing.T) {
		// Create a mock client
		mockClient := NewMockClient("test-model", "test-provider")

		// Create test middleware
		middleware1 := newTestMiddleware("middleware1")
		middleware2 := newTestMiddleware("middleware2")
		middlewares := []Middleware{middleware1, middleware2}

		// Wrap the client
		wrappedClient := ClientWithMiddleware(mockClient, middlewares)

		// Verify it returns an EnhancedClient
		enhancedClient, ok := wrappedClient.(*EnhancedClient)
		if !ok {
			t.Fatal("expected ClientWithMiddleware to return *EnhancedClient")
		}

		// Verify the original client is wrapped
		if enhancedClient.client != mockClient {
			t.Error("expected wrapped client to be the original mock client")
		}

		// Verify middleware were added
		names := enhancedClient.GetMiddlewareNames()
		expectedNames := []string{"middleware1", "middleware2"}
		if !equalStringSlice(names, expectedNames) {
			t.Errorf("expected middleware names %v, got %v", expectedNames, names)
		}
	})

	t.Run("adds middleware to existing EnhancedClient", func(t *testing.T) {
		// Create a mock client
		mockClient := NewMockClient("test-model", "test-provider")

		// Create initial enhanced client
		initialMiddleware := newTestMiddleware("initial")
		enhancedClient := NewEnhancedClient(mockClient, []Middleware{initialMiddleware})

		// Add more middleware using ClientWithMiddleware
		newMiddleware1 := newTestMiddleware("new1")
		newMiddleware2 := newTestMiddleware("new2")
		additionalMiddlewares := []Middleware{newMiddleware1, newMiddleware2}

		// Wrap the already enhanced client
		result := ClientWithMiddleware(enhancedClient, additionalMiddlewares)

		// Should return the same EnhancedClient instance
		if result != enhancedClient {
			t.Error("expected ClientWithMiddleware to return the same EnhancedClient instance")
		}

		// Verify all middleware are present
		names := enhancedClient.GetMiddlewareNames()
		expectedNames := []string{"initial", "new1", "new2"}
		if !equalStringSlice(names, expectedNames) {
			t.Errorf("expected middleware names %v, got %v", expectedNames, names)
		}
	})

	t.Run("works with empty middleware chain", func(t *testing.T) {
		// Create a mock client
		mockClient := NewMockClient("test-model", "test-provider")

		// Wrap with empty middleware chain
		wrappedClient := ClientWithMiddleware(mockClient, []Middleware{})

		// Should still return an EnhancedClient
		enhancedClient, ok := wrappedClient.(*EnhancedClient)
		if !ok {
			t.Fatal("expected ClientWithMiddleware to return *EnhancedClient even with empty middleware")
		}

		// Verify no middleware are present
		names := enhancedClient.GetMiddlewareNames()
		if len(names) != 0 {
			t.Errorf("expected empty middleware chain, got %v", names)
		}
	})

	t.Run("wrapped client functions correctly", func(t *testing.T) {
		// Create a mock client with expected response
		mockClient := NewMockClient("test-model", "test-provider")
		expectedResponse := ChatResponse{
			ID:    "test-123",
			Model: "test-model",
			Choices: []Choice{
				{
					Index: 0,
					Message: Message{
						Role:    RoleAssistant,
						Content: []MessageContent{NewTextContent("Hello, world!")},
					},
					FinishReason: "stop",
				},
			},
		}
		mockClient.AddResponse(expectedResponse)

		// Create middleware that modifies the model name
		modifyingMiddleware := newModifyingMiddleware("modifier", "-modified")

		// Wrap the client
		wrappedClient := ClientWithMiddleware(mockClient, []Middleware{modifyingMiddleware})

		// Test ChatCompletion
		ctx := context.Background()
		req := ChatRequest{
			Model:    "original-model",
			Messages: []Message{NewTextMessage(RoleUser, "Hello")},
		}

		resp, err := wrappedClient.ChatCompletion(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Verify the response was processed by middleware
		if resp.Model != "test-model-modified" {
			t.Errorf("expected model 'test-model-modified', got %s", resp.Model)
		}

		// Verify original response content is preserved
		if resp.ID != "test-123" {
			t.Errorf("expected ID 'test-123', got %s", resp.ID)
		}
	})

	t.Run("wrapped client preserves GetModelInfo", func(t *testing.T) {
		// Create a mock client
		mockClient := NewMockClient("test-model", "test-provider")

		// Wrap the client
		wrappedClient := ClientWithMiddleware(mockClient, []Middleware{newTestMiddleware("test")})

		// Test GetModelInfo
		modelInfo := wrappedClient.GetModelInfo()

		// Should return the original model info
		if modelInfo.Name != "test-model" {
			t.Errorf("expected model name 'test-model', got %s", modelInfo.Name)
		}
		if modelInfo.Provider != "test-provider" {
			t.Errorf("expected provider 'test-provider', got %s", modelInfo.Provider)
		}
	})

	t.Run("wrapped client handles Close correctly", func(t *testing.T) {
		// Create a mock client
		mockClient := NewMockClient("test-model", "test-provider")

		// Wrap the client
		wrappedClient := ClientWithMiddleware(mockClient, []Middleware{newTestMiddleware("test")})

		// Test Close
		err := wrappedClient.Close()
		if err != nil {
			t.Errorf("unexpected error from Close: %v", err)
		}
	})

	t.Run("middleware chain processes requests correctly", func(t *testing.T) {
		// Create a mock client
		mockClient := NewMockClient("test-model", "test-provider")
		mockClient.AddResponse(ChatResponse{
			ID:    "test-123",
			Model: "test-model",
			Choices: []Choice{
				{
					Index: 0,
					Message: Message{
						Role:    RoleAssistant,
						Content: []MessageContent{NewTextContent("Response")},
					},
					FinishReason: "stop",
				},
			},
		})

		// Create multiple modifying middleware
		middleware1 := newModifyingMiddleware("first", "-1")
		middleware2 := newModifyingMiddleware("second", "-2")

		// Wrap the client
		wrappedClient := ClientWithMiddleware(mockClient, []Middleware{middleware1, middleware2})

		ctx := context.Background()
		req := ChatRequest{
			Model:    "original",
			Messages: []Message{NewTextMessage(RoleUser, "Test")},
		}

		resp, err := wrappedClient.ChatCompletion(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Response should be processed by both middleware (in reverse order for responses)
		expectedModel := "test-model-2-1"
		if resp.Model != expectedModel {
			t.Errorf("expected model %s, got %s", expectedModel, resp.Model)
		}
	})

	t.Run("handles middleware errors in request processing", func(t *testing.T) {
		// Create a mock client
		mockClient := NewMockClient("test-model", "test-provider")

		// Create error middleware
		errorMiddleware := newErrorMiddleware("error-middleware")

		// Wrap the client
		wrappedClient := ClientWithMiddleware(mockClient, []Middleware{errorMiddleware})

		ctx := context.Background()
		req := ChatRequest{
			Model:    "test-model",
			Messages: []Message{NewTextMessage(RoleUser, "Test")},
		}

		_, err := wrappedClient.ChatCompletion(ctx, req)
		if err == nil {
			t.Fatal("expected error from middleware, but got none")
		}

		if !containsString(err.Error(), "middleware request processing failed") {
			t.Errorf("expected middleware error message, got: %v", err)
		}
	})
}

// TestClientWithMiddlewareStreaming tests streaming functionality with middleware
func TestClientWithMiddlewareStreaming(t *testing.T) {
	t.Run("streaming works with middleware", func(t *testing.T) {
		// Create a mock client
		mockClient := NewMockClient("test-model", "test-provider")

		// Set up streaming response
		streamEvents := []StreamEvent{
			{Type: "delta", Choice: &StreamChoice{Index: 0, Delta: &MessageDelta{Content: []MessageContent{NewTextContent("Hello")}}}},
			{Type: "delta", Choice: &StreamChoice{Index: 0, Delta: &MessageDelta{Content: []MessageContent{NewTextContent(" world")}}}},
			{Type: "done", Choice: &StreamChoice{Index: 0, FinishReason: "stop"}},
		}
		mockClient.WithStreamResponse(streamEvents)

		// Create modifying middleware
		modifyingMiddleware := newModifyingMiddleware("modifier", "-modified")

		// Wrap the client
		wrappedClient := ClientWithMiddleware(mockClient, []Middleware{modifyingMiddleware})

		ctx := context.Background()
		req := ChatRequest{
			Model:    "test-model",
			Messages: []Message{NewTextMessage(RoleUser, "Test")},
		}

		eventChan, err := wrappedClient.StreamChatCompletion(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Collect events
		var events []StreamEvent
		for event := range eventChan {
			events = append(events, event)
		}

		// Verify we got the expected number of events
		if len(events) != 3 {
			t.Fatalf("expected 3 events, got %d", len(events))
		}

		// Verify events were processed by middleware
		for _, event := range events {
			if event.Type == "delta-modified" || event.Type == "done-modified" {
				// Expected - middleware modified the type
			} else {
				t.Errorf("expected event type to be modified by middleware, got: %s", event.Type)
			}
		}
	})

	t.Run("streaming handles middleware errors gracefully", func(t *testing.T) {
		// Create a mock client
		mockClient := NewMockClient("test-model", "test-provider")

		// Set up streaming response
		streamEvents := []StreamEvent{
			{Type: "delta", Choice: &StreamChoice{Index: 0, Delta: &MessageDelta{Content: []MessageContent{NewTextContent("Hello")}}}},
		}
		mockClient.WithStreamResponse(streamEvents)

		// Create middleware that only fails on stream events, not requests
		streamErrorMiddleware := &mockMiddleware{
			name: "stream-error-middleware",
			reqMods: func(req *ChatRequest) (*ChatRequest, error) {
				return req, nil // Don't fail on requests
			},
			respMods: func(req *ChatRequest, resp *ChatResponse, err error) (*ChatResponse, error) {
				return resp, err // Don't fail on responses
			},
			eventMods: func(event StreamEvent) (StreamEvent, error) {
				return event, errors.New("stream event processing error") // Fail on stream events
			},
		}

		// Wrap the client
		wrappedClient := ClientWithMiddleware(mockClient, []Middleware{streamErrorMiddleware})

		ctx := context.Background()
		req := ChatRequest{
			Model:    "test-model",
			Messages: []Message{NewTextMessage(RoleUser, "Test")},
		}

		eventChan, err := wrappedClient.StreamChatCompletion(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should still get events (original events when middleware fails)
		var events []StreamEvent
		for event := range eventChan {
			events = append(events, event)
		}

		if len(events) == 0 {
			t.Error("expected to receive events even when middleware fails")
		}
	})
}

// Helper functions

func equalStringSlice(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func containsString(s, substr string) bool {
	return len(substr) <= len(s) && (substr == s || s[len(s)-len(substr):] == substr || s[:len(substr)] == substr || (len(s) > len(substr) && findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
