package llm

import (
	"context"
	"fmt"
	"sync"
)

// MessageHandler defines the interface for processing different types of message content.
// Handlers should be stateless and safe for concurrent use.
type MessageHandler interface {
	// CanHandle determines if this handler can process the given content
	CanHandle(content MessageContent) bool

	// Handle processes the content and returns modified content or nil if no response
	// Context should be properly propagated for cancellation/timeouts
	Handle(ctx context.Context, content MessageContent) (MessageContent, error)
}

// TypedMessageHandler is a concrete implementation of MessageHandler
// that handles a specific message type using a provided handler function.
type TypedMessageHandler struct {
	// SupportedType is the message type this handler supports
	SupportedType MessageType

	// Handler is the processing function
	Handler func(ctx context.Context, content MessageContent) (MessageContent, error)

	// mutex for thread safety (protecting the Handler function pointer)
	mu sync.RWMutex
}

// NewTypedMessageHandler creates a new TypedMessageHandler with the specified type and handler function.
// The handler function should validate input content type and handle context properly.
func NewTypedMessageHandler(msgType MessageType, handlerFunc func(ctx context.Context, content MessageContent) (MessageContent, error)) *TypedMessageHandler {
	if handlerFunc == nil {
		panic("handler function cannot be nil")
	}

	return &TypedMessageHandler{
		SupportedType: msgType,
		Handler:       handlerFunc,
	}
}

// CanHandle determines if this handler can process the given content.
// Returns true if the content type matches the handler's supported type.
func (h *TypedMessageHandler) CanHandle(content MessageContent) bool {
	if h == nil || content == nil {
		return false
	}

	h.mu.RLock()
	defer h.mu.RUnlock()

	return content.Type() == h.SupportedType
}

// Handle processes the content using the configured handler function.
// Returns nil content if no response is needed, or an error if processing fails.
func (h *TypedMessageHandler) Handle(ctx context.Context, content MessageContent) (MessageContent, error) {
	if h == nil {
		return nil, fmt.Errorf("handler is nil")
	}

	if content == nil {
		return nil, fmt.Errorf("content cannot be nil")
	}

	// Validate that we can handle this content type
	if !h.CanHandle(content) {
		return nil, fmt.Errorf("handler does not support content type: %s", content.Type())
	}

	h.mu.RLock()
	handlerFunc := h.Handler
	h.mu.RUnlock()

	if handlerFunc == nil {
		return nil, fmt.Errorf("handler function is nil")
	}

	// Call the handler function with proper context propagation
	return handlerFunc(ctx, content)
}

// HandlersForType returns all handlers that support the specified message type.
// Returns an empty slice if no handlers support the type.
func HandlersForType(msgType MessageType, handlers []MessageHandler) []MessageHandler {
	if handlers == nil {
		return []MessageHandler{}
	}

	var result []MessageHandler
	for _, handler := range handlers {
		if handler != nil {
			// Create dummy content to test handler capability
			var testContent MessageContent
			switch msgType {
			case MessageTypeText:
				testContent = NewTextContent("test")
			case MessageTypeImage:
				testContent = NewImageContentFromBytes([]byte{0x89, 0x50, 0x4E, 0x47}, "image/png")
			case MessageTypeFile:
				testContent = NewFileContentFromBytes([]byte("test"), "test.txt", "text/plain")
			default:
				continue
			}

			if testContent != nil && handler.CanHandle(testContent) {
				result = append(result, handler)
			}
		}
	}

	return result
}

// FindHandler returns the first handler that can process the given content.
// Returns nil if no suitable handler is found.
func FindHandler(content MessageContent, handlers []MessageHandler) MessageHandler {
	if content == nil || handlers == nil {
		return nil
	}

	for _, handler := range handlers {
		if handler != nil && handler.CanHandle(content) {
			return handler
		}
	}

	return nil
}
