package llm

import (
	"context"
	"fmt"
	"sync"
)

// MessageRouter provides a comprehensive routing system for dispatching
// different content types to appropriate handlers with thread-safe operations.
type MessageRouter struct {
	// handlers maps message types to their registered handlers
	handlers map[MessageType][]MessageHandler

	// defaultHandler provides fallback processing for unhandled types
	defaultHandler MessageHandler

	// mutex protects concurrent access to the router state
	mu sync.RWMutex
}

// NewMessageRouter creates a new MessageRouter instance with empty handler registry.
func NewMessageRouter() *MessageRouter {
	return &MessageRouter{
		handlers: make(map[MessageType][]MessageHandler),
	}
}

// RegisterHandler registers a handler for the specified message type.
// Multiple handlers can be registered for the same type (chain of responsibility).
// The handler is added to the end of the handler chain for that type.
func (r *MessageRouter) RegisterHandler(msgType MessageType, handler MessageHandler) {
	if handler == nil {
		return // Silently ignore nil handlers
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	// Initialize the slice if it doesn't exist
	if r.handlers[msgType] == nil {
		r.handlers[msgType] = make([]MessageHandler, 0)
	}

	r.handlers[msgType] = append(r.handlers[msgType], handler)
}

// UnregisterHandler removes a specific handler from the specified message type.
// Uses pointer equality to match handlers for removal.
// Returns true if the handler was found and removed, false otherwise.
func (r *MessageRouter) UnregisterHandler(msgType MessageType, handler MessageHandler) bool {
	if handler == nil {
		return false
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	handlersList, exists := r.handlers[msgType]
	if !exists {
		return false
	}

	// Find and remove the handler using pointer equality
	for i, h := range handlersList {
		if h == handler {
			// Remove handler by slicing around it
			r.handlers[msgType] = append(handlersList[:i], handlersList[i+1:]...)

			// Clean up empty slices
			if len(r.handlers[msgType]) == 0 {
				delete(r.handlers, msgType)
			}

			return true
		}
	}

	return false
}

// RouteMessage processes each content item in the message through appropriate handlers.
// For each content type, all registered handlers are executed in registration order.
// Results from all handlers are aggregated into a response message.
// Partial failures are collected and returned as a multi-error.
func (r *MessageRouter) RouteMessage(ctx context.Context, message Message) (Message, error) {
	if len(message.Content) == 0 {
		return Message{Role: RoleAssistant}, nil // Empty response for empty input
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	var responseContent []MessageContent
	var errors []error

	// Process each content item in the message
	for i, content := range message.Content {
		if content == nil {
			errors = append(errors, fmt.Errorf("content item %d is nil", i))
			continue
		}

		contentType := content.Type()

		// Get handlers for this content type
		handlers := r.handlers[contentType]

		// If no handlers found, try default handler
		if len(handlers) == 0 {
			if r.defaultHandler != nil && r.defaultHandler.CanHandle(content) {
				result, err := r.defaultHandler.Handle(ctx, content)
				if err != nil {
					errors = append(errors, fmt.Errorf("default handler failed for content %d: %w", i, err))
				} else if result != nil {
					responseContent = append(responseContent, result)
				}
			} else {
				errors = append(errors, fmt.Errorf("no handlers found for content type %s at index %d", contentType, i))
			}
			continue
		}

		// Execute all handlers for this content type
		for j, handler := range handlers {
			if handler == nil {
				errors = append(errors, fmt.Errorf("handler %d for type %s is nil", j, contentType))
				continue
			}

			if !handler.CanHandle(content) {
				continue // Skip handlers that can't process this content
			}

			result, err := handler.Handle(ctx, content)
			if err != nil {
				errors = append(errors, fmt.Errorf("handler %d failed for content %d (type %s): %w", j, i, contentType, err))
			} else if result != nil {
				responseContent = append(responseContent, result)
			}
		}
	}

	// Create response message
	response := Message{
		Role:    RoleAssistant,
		Content: responseContent,
	}

	// If we have any metadata from the input, preserve it
	if message.Metadata != nil {
		response.Metadata = make(map[string]any)
		for k, v := range message.Metadata {
			response.Metadata[k] = v
		}
	}

	// Return aggregated errors if any occurred
	if len(errors) > 0 {
		return response, &MultiError{Errors: errors}
	}

	return response, nil
}

// GetHandlers returns a copy of all handlers registered for the specified message type.
// Returns an empty slice if no handlers are registered for the type.
func (r *MessageRouter) GetHandlers(msgType MessageType) []MessageHandler {
	r.mu.RLock()
	defer r.mu.RUnlock()

	handlersList, exists := r.handlers[msgType]
	if !exists {
		return []MessageHandler{}
	}

	// Return a copy to prevent external modification
	result := make([]MessageHandler, len(handlersList))
	copy(result, handlersList)
	return result
}

// SetDefaultHandler sets the fallback handler for unhandled content types.
// The default handler is used when no specific handlers are registered for a content type.
// Pass nil to remove the default handler.
func (r *MessageRouter) SetDefaultHandler(handler MessageHandler) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.defaultHandler = handler
}

// GetDefaultHandler returns the current default handler.
// Returns nil if no default handler is set.
func (r *MessageRouter) GetDefaultHandler() MessageHandler {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return r.defaultHandler
}

// SupportedTypes returns a slice of all message types that have registered handlers.
// The order of types is not guaranteed.
func (r *MessageRouter) SupportedTypes() []MessageType {
	r.mu.RLock()
	defer r.mu.RUnlock()

	types := make([]MessageType, 0, len(r.handlers))
	for msgType := range r.handlers {
		types = append(types, msgType)
	}

	return types
}

// Clear removes all registered handlers and the default handler.
// This operation is thread-safe and atomic.
func (r *MessageRouter) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Clear all handlers
	r.handlers = make(map[MessageType][]MessageHandler)
	r.defaultHandler = nil
}

// HandlerCount returns the total number of handlers registered across all types.
func (r *MessageRouter) HandlerCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	count := 0
	for _, handlers := range r.handlers {
		count += len(handlers)
	}

	return count
}

// HasHandlersForType checks if any handlers are registered for the specified type.
func (r *MessageRouter) HasHandlersForType(msgType MessageType) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	handlers, exists := r.handlers[msgType]
	return exists && len(handlers) > 0
}

// MultiError represents multiple errors that occurred during message routing.
type MultiError struct {
	Errors []error
}

// Error implements the error interface by joining all error messages.
func (me *MultiError) Error() string {
	if len(me.Errors) == 0 {
		return ""
	}

	if len(me.Errors) == 1 {
		return me.Errors[0].Error()
	}

	var result string
	for i, err := range me.Errors {
		if i > 0 {
			result += "; "
		}
		result += err.Error()
	}

	return result
}

// Unwrap returns the underlying errors for error inspection.
func (me *MultiError) Unwrap() []error {
	return me.Errors
}
