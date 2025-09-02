package llm

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestTypedMessageHandler_NewTypedMessageHandler(t *testing.T) {
	tests := []struct {
		name        string
		msgType     MessageType
		handlerFunc func(ctx context.Context, content MessageContent) (MessageContent, error)
		shouldPanic bool
	}{
		{
			name:    "Valid text handler",
			msgType: MessageTypeText,
			handlerFunc: func(ctx context.Context, content MessageContent) (MessageContent, error) {
				return content, nil
			},
			shouldPanic: false,
		},
		{
			name:    "Valid image handler",
			msgType: MessageTypeImage,
			handlerFunc: func(ctx context.Context, content MessageContent) (MessageContent, error) {
				return NewTextContent("Processed image"), nil
			},
			shouldPanic: false,
		},
		{
			name:        "Nil handler function should panic",
			msgType:     MessageTypeText,
			handlerFunc: nil,
			shouldPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.shouldPanic {
				defer func() {
					if r := recover(); r == nil {
						t.Errorf("NewTypedMessageHandler() should have panicked with nil handler function")
					}
				}()
			}

			handler := NewTypedMessageHandler(tt.msgType, tt.handlerFunc)

			if !tt.shouldPanic {
				if handler == nil {
					t.Errorf("NewTypedMessageHandler() returned nil")
				} else if handler.SupportedType != tt.msgType {
					t.Errorf("NewTypedMessageHandler() SupportedType = %v, want %v", handler.SupportedType, tt.msgType)
				}
			}
		})
	}
}

func TestTypedMessageHandler_CanHandle(t *testing.T) {
	textContent := NewTextContent("test message")
	imageContent := NewImageContentFromBytes([]byte{0x89, 0x50, 0x4E, 0x47}, "image/png")
	fileContent := NewFileContentFromBytes([]byte("test"), "test.txt", "text/plain")

	tests := []struct {
		name     string
		handler  *TypedMessageHandler
		content  MessageContent
		expected bool
	}{
		{
			name:     "Text handler can handle text content",
			handler:  NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) { return content, nil }),
			content:  textContent,
			expected: true,
		},
		{
			name:     "Text handler cannot handle image content",
			handler:  NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) { return content, nil }),
			content:  imageContent,
			expected: false,
		},
		{
			name:     "Image handler can handle image content",
			handler:  NewTypedMessageHandler(MessageTypeImage, func(ctx context.Context, content MessageContent) (MessageContent, error) { return content, nil }),
			content:  imageContent,
			expected: true,
		},
		{
			name:     "File handler can handle file content",
			handler:  NewTypedMessageHandler(MessageTypeFile, func(ctx context.Context, content MessageContent) (MessageContent, error) { return content, nil }),
			content:  fileContent,
			expected: true,
		},
		{
			name:     "Nil handler returns false",
			handler:  nil,
			content:  textContent,
			expected: false,
		},
		{
			name:     "Nil content returns false",
			handler:  NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) { return content, nil }),
			content:  nil,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.handler.CanHandle(tt.content)
			if result != tt.expected {
				t.Errorf("CanHandle() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestTypedMessageHandler_Handle(t *testing.T) {
	textContent := NewTextContent("test message")
	imageContent := NewImageContentFromBytes([]byte{0x89, 0x50, 0x4E, 0x47}, "image/png")

	tests := []struct {
		name        string
		handler     *TypedMessageHandler
		content     MessageContent
		expectError bool
		errorMsg    string
	}{
		{
			name: "Successful handling",
			handler: NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
				return NewTextContent("Processed: " + content.(*TextContent).Text), nil
			}),
			content:     textContent,
			expectError: false,
		},
		{
			name: "Handler function returns error",
			handler: NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
				return nil, errors.New("processing failed")
			}),
			content:     textContent,
			expectError: true,
			errorMsg:    "processing failed",
		},
		{
			name: "Handler function returns nil content",
			handler: NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
				return nil, nil
			}),
			content:     textContent,
			expectError: false,
		},
		{
			name:        "Nil handler returns error",
			handler:     nil,
			content:     textContent,
			expectError: true,
			errorMsg:    "handler is nil",
		},
		{
			name: "Nil content returns error",
			handler: NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
				return content, nil
			}),
			content:     nil,
			expectError: true,
			errorMsg:    "content cannot be nil",
		},
		{
			name: "Wrong content type returns error",
			handler: NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
				return content, nil
			}),
			content:     imageContent,
			expectError: true,
			errorMsg:    "handler does not support content type: image",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			result, err := tt.handler.Handle(ctx, tt.content)

			if tt.expectError {
				if err == nil {
					t.Errorf("Handle() expected error but got none")
				}
				if tt.errorMsg != "" && err.Error() != tt.errorMsg {
					t.Errorf("Handle() error = %v, want %v", err.Error(), tt.errorMsg)
				}
			} else {
				if err != nil {
					t.Errorf("Handle() unexpected error = %v", err)
				}
				// For successful cases, we can check the result if needed
				_ = result
			}
		})
	}
}

func TestTypedMessageHandler_ContextHandling(t *testing.T) {
	textContent := NewTextContent("test message")

	t.Run("Context cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())

		handler := NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(100 * time.Millisecond):
				return content, nil
			}
		})

		cancel() // Cancel before handling

		_, err := handler.Handle(ctx, textContent)
		if err == nil {
			t.Errorf("Handle() expected context cancellation error")
		}
		if err != context.Canceled {
			t.Errorf("Handle() error = %v, want %v", err, context.Canceled)
		}
	})

	t.Run("Context timeout", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		defer cancel()

		handler := NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(100 * time.Millisecond):
				return content, nil
			}
		})

		_, err := handler.Handle(ctx, textContent)
		if err == nil {
			t.Errorf("Handle() expected timeout error")
		}
		if err != context.DeadlineExceeded {
			t.Errorf("Handle() error = %v, want %v", err, context.DeadlineExceeded)
		}
	})
}

func TestTypedMessageHandler_ConcurrentAccess(t *testing.T) {
	handler := NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
		time.Sleep(10 * time.Millisecond) // Simulate some processing
		return NewTextContent("Processed: " + content.(*TextContent).Text), nil
	})

	const numGoroutines = 50
	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)
	results := make(chan MessageContent, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ctx := context.Background()

			content := NewTextContent(fmt.Sprintf("message-%d", id))
			result, err := handler.Handle(ctx, content)

			if err != nil {
				errors <- err
				return
			}
			results <- result
		}(i)
	}

	wg.Wait()
	close(errors)
	close(results)

	// Check for errors
	for err := range errors {
		t.Errorf("Concurrent access error: %v", err)
	}

	// Check results count
	resultCount := 0
	for range results {
		resultCount++
	}

	if resultCount != numGoroutines {
		t.Errorf("Expected %d results, got %d", numGoroutines, resultCount)
	}
}

func TestHandlersForType(t *testing.T) {
	textHandler1 := NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) { return content, nil })
	textHandler2 := NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) { return content, nil })
	imageHandler := NewTypedMessageHandler(MessageTypeImage, func(ctx context.Context, content MessageContent) (MessageContent, error) { return content, nil })
	fileHandler := NewTypedMessageHandler(MessageTypeFile, func(ctx context.Context, content MessageContent) (MessageContent, error) { return content, nil })

	tests := []struct {
		name          string
		msgType       MessageType
		handlers      []MessageHandler
		expectedCount int
	}{
		{
			name:          "Find text handlers",
			msgType:       MessageTypeText,
			handlers:      []MessageHandler{textHandler1, imageHandler, textHandler2, fileHandler},
			expectedCount: 2,
		},
		{
			name:          "Find image handlers",
			msgType:       MessageTypeImage,
			handlers:      []MessageHandler{textHandler1, imageHandler, fileHandler},
			expectedCount: 1,
		},
		{
			name:          "Find file handlers",
			msgType:       MessageTypeFile,
			handlers:      []MessageHandler{textHandler1, imageHandler, fileHandler},
			expectedCount: 1,
		},
		{
			name:          "No handlers found",
			msgType:       MessageTypeText,
			handlers:      []MessageHandler{imageHandler, fileHandler},
			expectedCount: 0,
		},
		{
			name:          "Nil handlers slice",
			msgType:       MessageTypeText,
			handlers:      nil,
			expectedCount: 0,
		},
		{
			name:          "Empty handlers slice",
			msgType:       MessageTypeText,
			handlers:      []MessageHandler{},
			expectedCount: 0,
		},
		{
			name:          "Handlers with nil entries",
			msgType:       MessageTypeText,
			handlers:      []MessageHandler{textHandler1, nil, textHandler2, nil},
			expectedCount: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := HandlersForType(tt.msgType, tt.handlers)
			if len(result) != tt.expectedCount {
				t.Errorf("HandlersForType() returned %d handlers, want %d", len(result), tt.expectedCount)
			}
		})
	}
}

func TestFindHandler(t *testing.T) {
	textContent := NewTextContent("test message")
	imageContent := NewImageContentFromBytes([]byte{0x89, 0x50, 0x4E, 0x47}, "image/png")
	fileContent := NewFileContentFromBytes([]byte("test"), "test.txt", "text/plain")

	textHandler := NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) { return content, nil })
	imageHandler := NewTypedMessageHandler(MessageTypeImage, func(ctx context.Context, content MessageContent) (MessageContent, error) { return content, nil })
	fileHandler := NewTypedMessageHandler(MessageTypeFile, func(ctx context.Context, content MessageContent) (MessageContent, error) { return content, nil })

	tests := []struct {
		name     string
		content  MessageContent
		handlers []MessageHandler
		expected MessageHandler
	}{
		{
			name:     "Find text handler",
			content:  textContent,
			handlers: []MessageHandler{textHandler, imageHandler, fileHandler},
			expected: textHandler,
		},
		{
			name:     "Find image handler",
			content:  imageContent,
			handlers: []MessageHandler{textHandler, imageHandler, fileHandler},
			expected: imageHandler,
		},
		{
			name:     "Find file handler",
			content:  fileContent,
			handlers: []MessageHandler{textHandler, imageHandler, fileHandler},
			expected: fileHandler,
		},
		{
			name:     "No suitable handler found",
			content:  textContent,
			handlers: []MessageHandler{imageHandler, fileHandler},
			expected: nil,
		},
		{
			name:     "Nil content",
			content:  nil,
			handlers: []MessageHandler{textHandler, imageHandler, fileHandler},
			expected: nil,
		},
		{
			name:     "Nil handlers slice",
			content:  textContent,
			handlers: nil,
			expected: nil,
		},
		{
			name:     "Empty handlers slice",
			content:  textContent,
			handlers: []MessageHandler{},
			expected: nil,
		},
		{
			name:     "Handlers with nil entries",
			content:  textContent,
			handlers: []MessageHandler{nil, textHandler, nil},
			expected: textHandler,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FindHandler(tt.content, tt.handlers)
			if result != tt.expected {
				t.Errorf("FindHandler() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestMessageHandlerInterface(t *testing.T) {
	// Verify that TypedMessageHandler implements MessageHandler interface
	var _ MessageHandler = &TypedMessageHandler{}

	// Test with actual implementation
	handler := NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
		return content, nil
	})

	// Verify interface methods work
	textContent := NewTextContent("test")

	canHandle := handler.CanHandle(textContent)
	if !canHandle {
		t.Errorf("Handler should be able to handle text content")
	}

	ctx := context.Background()
	result, err := handler.Handle(ctx, textContent)
	if err != nil {
		t.Errorf("Handler should not return error: %v", err)
	}
	if result == nil {
		t.Errorf("Handler should return result")
	}
}

func TestEdgeCases(t *testing.T) {
	t.Run("Handler with nil function after creation", func(t *testing.T) {
		handler := NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
			return content, nil
		})

		// Simulate the unlikely case of handler function becoming nil
		handler.mu.Lock()
		handler.Handler = nil
		handler.mu.Unlock()

		textContent := NewTextContent("test")
		_, err := handler.Handle(context.Background(), textContent)
		if err == nil || err.Error() != "handler function is nil" {
			t.Errorf("Expected 'handler function is nil' error, got: %v", err)
		}
	})

	t.Run("Invalid message type in HandlersForType", func(t *testing.T) {
		handler := NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
			return content, nil
		})

		result := HandlersForType("invalid_type", []MessageHandler{handler})
		if len(result) != 0 {
			t.Errorf("HandlersForType should return empty slice for invalid type")
		}
	})

	t.Run("Handler validates input content type", func(t *testing.T) {
		handler := NewTypedMessageHandler(MessageTypeText, func(ctx context.Context, content MessageContent) (MessageContent, error) {
			// Handler function should validate content type
			if content.Type() != MessageTypeText {
				return nil, errors.New("invalid content type")
			}
			return content, nil
		})

		imageContent := NewImageContentFromBytes([]byte{0x89, 0x50, 0x4E, 0x47}, "image/png")

		_, err := handler.Handle(context.Background(), imageContent)
		if err == nil {
			t.Errorf("Handler should reject wrong content type")
		}
		if err.Error() != "handler does not support content type: image" {
			t.Errorf("Expected type mismatch error, got: %v", err)
		}
	})
}
