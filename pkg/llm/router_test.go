package llm

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestMessageHandler is a mock handler for testing
type TestMessageHandler struct {
	SupportedType MessageType
	ProcessFunc   func(ctx context.Context, content MessageContent) (MessageContent, error)
	HandleCount   int
	mu            sync.Mutex
}

func NewTestHandler(msgType MessageType, processFunc func(ctx context.Context, content MessageContent) (MessageContent, error)) *TestMessageHandler {
	return &TestMessageHandler{
		SupportedType: msgType,
		ProcessFunc:   processFunc,
	}
}

func (h *TestMessageHandler) CanHandle(content MessageContent) bool {
	if h == nil || content == nil {
		return false
	}
	return content.Type() == h.SupportedType
}

func (h *TestMessageHandler) Handle(ctx context.Context, content MessageContent) (MessageContent, error) {
	h.mu.Lock()
	h.HandleCount++
	h.mu.Unlock()

	if h.ProcessFunc != nil {
		return h.ProcessFunc(ctx, content)
	}

	// Default behavior: echo the content
	return content, nil
}

func (h *TestMessageHandler) GetHandleCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.HandleCount
}

func TestNewMessageRouter(t *testing.T) {
	router := NewMessageRouter()

	if router == nil {
		t.Fatal("NewMessageRouter returned nil")
	}

	if router.handlers == nil {
		t.Error("handlers map not initialized")
	}

	if len(router.handlers) != 0 {
		t.Error("expected empty handlers map")
	}

	if router.defaultHandler != nil {
		t.Error("expected nil default handler")
	}
}

func TestRegisterHandler(t *testing.T) {
	router := NewMessageRouter()

	// Test registering single handler
	handler1 := NewTestHandler(MessageTypeText, nil)
	router.RegisterHandler(MessageTypeText, handler1)

	handlers := router.GetHandlers(MessageTypeText)
	if len(handlers) != 1 {
		t.Errorf("expected 1 handler, got %d", len(handlers))
	}

	// Test registering multiple handlers for same type
	handler2 := NewTestHandler(MessageTypeText, nil)
	router.RegisterHandler(MessageTypeText, handler2)

	handlers = router.GetHandlers(MessageTypeText)
	if len(handlers) != 2 {
		t.Errorf("expected 2 handlers, got %d", len(handlers))
	}

	// Test registering handlers for different types
	imageHandler := NewTestHandler(MessageTypeImage, nil)
	router.RegisterHandler(MessageTypeImage, imageHandler)

	if len(router.GetHandlers(MessageTypeImage)) != 1 {
		t.Error("expected 1 image handler")
	}

	// Test registering nil handler (should be ignored)
	initialCount := len(router.GetHandlers(MessageTypeText))
	router.RegisterHandler(MessageTypeText, nil)

	if len(router.GetHandlers(MessageTypeText)) != initialCount {
		t.Error("nil handler should be ignored")
	}
}

func TestUnregisterHandler(t *testing.T) {
	router := NewMessageRouter()

	handler1 := NewTestHandler(MessageTypeText, nil)
	handler2 := NewTestHandler(MessageTypeText, nil)

	router.RegisterHandler(MessageTypeText, handler1)
	router.RegisterHandler(MessageTypeText, handler2)

	// Test successful unregistration
	if !router.UnregisterHandler(MessageTypeText, handler1) {
		t.Error("expected successful unregistration")
	}

	handlers := router.GetHandlers(MessageTypeText)
	if len(handlers) != 1 {
		t.Errorf("expected 1 handler after unregistration, got %d", len(handlers))
	}

	// Test unregistering non-existent handler
	fakeHandler := NewTestHandler(MessageTypeText, nil)
	if router.UnregisterHandler(MessageTypeText, fakeHandler) {
		t.Error("should not unregister non-existent handler")
	}

	// Test unregistering from non-existent type
	if router.UnregisterHandler(MessageTypeFile, handler1) {
		t.Error("should not unregister from non-existent type")
	}

	// Test unregistering nil handler
	if router.UnregisterHandler(MessageTypeText, nil) {
		t.Error("should not unregister nil handler")
	}

	// Test cleanup of empty handler lists
	router.UnregisterHandler(MessageTypeText, handler2)
	if router.HasHandlersForType(MessageTypeText) {
		t.Error("handler type should be cleaned up when empty")
	}
}

func TestRouteMessage_SingleContentType(t *testing.T) {
	router := NewMessageRouter()
	ctx := context.Background()

	// Register handler that uppercases text
	handler := NewTestHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
		if textContent, ok := content.(*TextContent); ok {
			return NewTextContent("PROCESSED: " + textContent.GetText()), nil
		}
		return nil, errors.New("expected text content")
	})

	router.RegisterHandler(MessageTypeText, handler)

	// Create test message
	message := Message{
		Role:    RoleUser,
		Content: []MessageContent{NewTextContent("hello world")},
	}

	// Route message
	response, err := router.RouteMessage(ctx, message)
	if err != nil {
		t.Fatalf("routing failed: %v", err)
	}

	if len(response.Content) != 1 {
		t.Fatalf("expected 1 content item, got %d", len(response.Content))
	}

	if textContent, ok := response.Content[0].(*TextContent); ok {
		expected := "PROCESSED: hello world"
		if textContent.GetText() != expected {
			t.Errorf("expected %q, got %q", expected, textContent.GetText())
		}
	} else {
		t.Error("expected text content in response")
	}

	if response.Role != RoleAssistant {
		t.Errorf("expected assistant role, got %v", response.Role)
	}
}

func TestRouteMessage_MultipleContentTypes(t *testing.T) {
	router := NewMessageRouter()
	ctx := context.Background()

	// Register handlers for different types
	textHandler := NewTestHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
		return NewTextContent("text-processed"), nil
	})

	imageHandler := NewTestHandler(MessageTypeImage, func(ctx context.Context, content MessageContent) (MessageContent, error) {
		return NewTextContent("image-processed"), nil
	})

	router.RegisterHandler(MessageTypeText, textHandler)
	router.RegisterHandler(MessageTypeImage, imageHandler)

	// Create message with mixed content
	message := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("some text"),
			NewImageContentFromBytes([]byte{0x89, 0x50, 0x4E, 0x47}, "image/png"),
		},
	}

	response, err := router.RouteMessage(ctx, message)
	if err != nil {
		t.Fatalf("routing failed: %v", err)
	}

	if len(response.Content) != 2 {
		t.Fatalf("expected 2 content items, got %d", len(response.Content))
	}

	// Verify both handlers were called
	if textHandler.GetHandleCount() != 1 {
		t.Error("text handler should be called once")
	}
	if imageHandler.GetHandleCount() != 1 {
		t.Error("image handler should be called once")
	}
}

func TestRouteMessage_HandlerChain(t *testing.T) {
	router := NewMessageRouter()
	ctx := context.Background()

	// Register multiple handlers for the same type
	handler1 := NewTestHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
		return NewTextContent("handler1"), nil
	})

	handler2 := NewTestHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
		return NewTextContent("handler2"), nil
	})

	router.RegisterHandler(MessageTypeText, handler1)
	router.RegisterHandler(MessageTypeText, handler2)

	message := Message{
		Role:    RoleUser,
		Content: []MessageContent{NewTextContent("test")},
	}

	response, err := router.RouteMessage(ctx, message)
	if err != nil {
		t.Fatalf("routing failed: %v", err)
	}

	// Both handlers should execute and produce results
	if len(response.Content) != 2 {
		t.Fatalf("expected 2 content items from chain, got %d", len(response.Content))
	}

	// Verify execution order
	if textContent1, ok := response.Content[0].(*TextContent); ok {
		if textContent1.GetText() != "handler1" {
			t.Error("first handler result incorrect")
		}
	}

	if textContent2, ok := response.Content[1].(*TextContent); ok {
		if textContent2.GetText() != "handler2" {
			t.Error("second handler result incorrect")
		}
	}
}

func TestRouteMessage_DefaultHandler(t *testing.T) {
	router := NewMessageRouter()
	ctx := context.Background()

	// Set default handler
	defaultHandler := NewTestHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
		return NewTextContent("default-handled"), nil
	})

	router.SetDefaultHandler(defaultHandler)

	// Create message with unhandled content type
	message := Message{
		Role:    RoleUser,
		Content: []MessageContent{NewTextContent("unhandled")},
	}

	response, err := router.RouteMessage(ctx, message)
	if err != nil {
		t.Fatalf("routing failed: %v", err)
	}

	if len(response.Content) != 1 {
		t.Fatalf("expected 1 content item, got %d", len(response.Content))
	}

	if textContent, ok := response.Content[0].(*TextContent); ok {
		if textContent.GetText() != "default-handled" {
			t.Error("default handler not used")
		}
	}
}

func TestRouteMessage_ErrorHandling(t *testing.T) {
	router := NewMessageRouter()
	ctx := context.Background()

	// Register handler that returns error
	errorHandler := NewTestHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
		return nil, errors.New("handler error")
	})

	router.RegisterHandler(MessageTypeText, errorHandler)

	message := Message{
		Role:    RoleUser,
		Content: []MessageContent{NewTextContent("test")},
	}

	response, err := router.RouteMessage(ctx, message)
	if err == nil {
		t.Fatal("expected error from routing")
	}

	// Check for MultiError
	var multiErr *MultiError
	if !errors.As(err, &multiErr) {
		t.Error("expected MultiError type")
	}

	if len(multiErr.Errors) != 1 {
		t.Errorf("expected 1 error, got %d", len(multiErr.Errors))
	}

	// Response should still be valid
	if response.Role != RoleAssistant {
		t.Error("response should have assistant role")
	}
}

func TestRouteMessage_PartialFailures(t *testing.T) {
	router := NewMessageRouter()
	ctx := context.Background()

	// Register one successful and one failing handler
	successHandler := NewTestHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
		return NewTextContent("success"), nil
	})

	failHandler := NewTestHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
		return nil, errors.New("fail")
	})

	router.RegisterHandler(MessageTypeText, successHandler)
	router.RegisterHandler(MessageTypeText, failHandler)

	message := Message{
		Role:    RoleUser,
		Content: []MessageContent{NewTextContent("test")},
	}

	response, err := router.RouteMessage(ctx, message)
	if err == nil {
		t.Fatal("expected error from partial failure")
	}

	// Should have one successful result
	if len(response.Content) != 1 {
		t.Errorf("expected 1 successful result, got %d", len(response.Content))
	}

	if textContent, ok := response.Content[0].(*TextContent); ok {
		if textContent.GetText() != "success" {
			t.Error("successful result missing")
		}
	}
}

func TestRouteMessage_EdgeCases(t *testing.T) {
	router := NewMessageRouter()
	ctx := context.Background()

	t.Run("empty message", func(t *testing.T) {
		message := Message{Role: RoleUser}
		response, err := router.RouteMessage(ctx, message)
		if err != nil {
			t.Errorf("empty message should not error: %v", err)
		}
		if len(response.Content) != 0 {
			t.Error("empty message should produce empty response")
		}
	})

	t.Run("nil content", func(t *testing.T) {
		message := Message{
			Role:    RoleUser,
			Content: []MessageContent{nil},
		}
		_, err := router.RouteMessage(ctx, message)
		if err == nil {
			t.Error("nil content should produce error")
		}

		var multiErr *MultiError
		if errors.As(err, &multiErr) {
			if len(multiErr.Errors) != 1 {
				t.Error("expected 1 error for nil content")
			}
		}
	})

	t.Run("no handlers", func(t *testing.T) {
		message := Message{
			Role:    RoleUser,
			Content: []MessageContent{NewTextContent("test")},
		}
		response, err := router.RouteMessage(ctx, message)
		if err == nil {
			t.Error("no handlers should produce error")
		}

		if len(response.Content) != 0 {
			t.Error("no handlers should produce empty response")
		}
	})
}

func TestRouteMessage_MetadataPreservation(t *testing.T) {
	router := NewMessageRouter()
	ctx := context.Background()

	handler := NewTestHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
		return NewTextContent("processed"), nil
	})

	router.RegisterHandler(MessageTypeText, handler)

	message := Message{
		Role:     RoleUser,
		Content:  []MessageContent{NewTextContent("test")},
		Metadata: map[string]any{"key": "value"},
	}

	response, err := router.RouteMessage(ctx, message)
	if err != nil {
		t.Fatalf("routing failed: %v", err)
	}

	if response.Metadata == nil {
		t.Fatal("metadata should be preserved")
	}

	if value, exists := response.Metadata["key"]; !exists || value != "value" {
		t.Error("metadata not properly preserved")
	}
}

func TestConcurrentOperations(t *testing.T) {
	router := NewMessageRouter()
	ctx := context.Background()

	const numGoroutines = 50
	const numOperations = 100

	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines*numOperations)

	// Concurrent handler registration
	wg.Add(numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				handler := NewTestHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
					return content, nil
				})
				router.RegisterHandler(MessageTypeText, handler)
			}
		}(i)
	}

	// Concurrent routing
	wg.Add(numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				message := Message{
					Role:    RoleUser,
					Content: []MessageContent{NewTextContent(fmt.Sprintf("test-%d-%d", id, j))},
				}
				if _, err := router.RouteMessage(ctx, message); err != nil {
					errors <- err
				}
			}
		}(i)
	}

	// Concurrent handler unregistration
	wg.Add(numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				handlers := router.GetHandlers(MessageTypeText)
				if len(handlers) > 0 {
					// Try to unregister a random handler
					router.UnregisterHandler(MessageTypeText, handlers[0])
				}
			}
		}(i)
	}

	wg.Wait()
	close(errors)

	// Check for any errors
	for err := range errors {
		t.Errorf("concurrent operation error: %v", err)
	}
}

func TestPerformanceWithMultipleHandlers(t *testing.T) {
	router := NewMessageRouter()
	ctx := context.Background()

	// Register many handlers
	const numHandlers = 100
	for i := 0; i < numHandlers; i++ {
		handler := NewTestHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
			// Simulate some work
			time.Sleep(time.Microsecond)
			return content, nil
		})
		router.RegisterHandler(MessageTypeText, handler)
	}

	// Create large message
	const numContentItems = 50
	message := Message{Role: RoleUser}
	for i := 0; i < numContentItems; i++ {
		message.Content = append(message.Content, NewTextContent(fmt.Sprintf("content-%d", i)))
	}

	start := time.Now()
	response, err := router.RouteMessage(ctx, message)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("performance test failed: %v", err)
	}

	expectedResults := numHandlers * numContentItems
	if len(response.Content) != expectedResults {
		t.Errorf("expected %d results, got %d", expectedResults, len(response.Content))
	}

	t.Logf("Performance test: %d handlers × %d content items processed in %v",
		numHandlers, numContentItems, duration)

	// Reasonable performance check (adjust threshold as needed)
	if duration > time.Second*5 {
		t.Errorf("performance test took too long: %v", duration)
	}
}

func TestUtilityMethods(t *testing.T) {
	router := NewMessageRouter()

	t.Run("SupportedTypes", func(t *testing.T) {
		if len(router.SupportedTypes()) != 0 {
			t.Error("new router should have no supported types")
		}

		router.RegisterHandler(MessageTypeText, NewTestHandler(MessageTypeText, nil))
		router.RegisterHandler(MessageTypeImage, NewTestHandler(MessageTypeImage, nil))

		types := router.SupportedTypes()
		if len(types) != 2 {
			t.Errorf("expected 2 supported types, got %d", len(types))
		}
	})

	t.Run("HandlerCount", func(t *testing.T) {
		router.Clear()
		if router.HandlerCount() != 0 {
			t.Error("cleared router should have 0 handlers")
		}

		router.RegisterHandler(MessageTypeText, NewTestHandler(MessageTypeText, nil))
		router.RegisterHandler(MessageTypeText, NewTestHandler(MessageTypeText, nil))
		router.RegisterHandler(MessageTypeImage, NewTestHandler(MessageTypeImage, nil))

		if router.HandlerCount() != 3 {
			t.Errorf("expected 3 handlers, got %d", router.HandlerCount())
		}
	})

	t.Run("HasHandlersForType", func(t *testing.T) {
		router.Clear()
		if router.HasHandlersForType(MessageTypeText) {
			t.Error("should not have handlers for text")
		}

		router.RegisterHandler(MessageTypeText, NewTestHandler(MessageTypeText, nil))
		if !router.HasHandlersForType(MessageTypeText) {
			t.Error("should have handlers for text")
		}
	})

	t.Run("Clear", func(t *testing.T) {
		router.RegisterHandler(MessageTypeText, NewTestHandler(MessageTypeText, nil))
		router.SetDefaultHandler(NewTestHandler(MessageTypeText, nil))

		router.Clear()

		if len(router.SupportedTypes()) != 0 {
			t.Error("clear should remove all handlers")
		}
		if router.GetDefaultHandler() != nil {
			t.Error("clear should remove default handler")
		}
	})

	t.Run("DefaultHandler", func(t *testing.T) {
		if router.GetDefaultHandler() != nil {
			t.Error("initial default handler should be nil")
		}

		handler := NewTestHandler(MessageTypeText, nil)
		router.SetDefaultHandler(handler)

		if router.GetDefaultHandler() != handler {
			t.Error("default handler not set correctly")
		}

		router.SetDefaultHandler(nil)
		if router.GetDefaultHandler() != nil {
			t.Error("default handler not cleared")
		}
	})
}

func TestMultiError(t *testing.T) {
	t.Run("single error", func(t *testing.T) {
		err := errors.New("test error")
		multiErr := &MultiError{Errors: []error{err}}

		if multiErr.Error() != "test error" {
			t.Errorf("expected 'test error', got %q", multiErr.Error())
		}

		unwrapped := multiErr.Unwrap()
		if len(unwrapped) != 1 || unwrapped[0] != err {
			t.Error("unwrap failed for single error")
		}
	})

	t.Run("multiple errors", func(t *testing.T) {
		err1 := errors.New("error 1")
		err2 := errors.New("error 2")
		multiErr := &MultiError{Errors: []error{err1, err2}}

		expected := "error 1; error 2"
		if multiErr.Error() != expected {
			t.Errorf("expected %q, got %q", expected, multiErr.Error())
		}
	})

	t.Run("empty errors", func(t *testing.T) {
		multiErr := &MultiError{Errors: []error{}}
		if multiErr.Error() != "" {
			t.Error("empty errors should return empty string")
		}
	})
}
