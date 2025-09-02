// Package llm provides abstractions for Large Language Model clients
// stream_utils.go defines utilities for stream conversion and compatibility

package llm

import "context"

// ToolStreamEvent interface for compatibility with external tool streaming systems
type ToolStreamEvent interface {
	GetToolName() string
	GetStatus() string
	GetContent() string
	GetProgress() *ToolProgressInfo
	GetError() *ToolExecutionError
	GetMetadata() map[string]interface{}
}

// ConvertToolStreamToLLMStream converts external ToolStreamEvent to LLM StreamEvent
func ConvertToolStreamToLLMStream(toolEvent ToolStreamEvent, toolCallID string) StreamEvent {
	switch toolEvent.GetStatus() {
	case "start":
		return NewToolStartEvent(toolEvent.GetToolName(), toolCallID, toolEvent.GetMetadata())
	case "progress":
		return NewToolProgressEvent(toolEvent.GetToolName(), toolCallID, toolEvent.GetProgress())
	case "data":
		return NewToolDataEvent(toolEvent.GetToolName(), toolCallID, toolEvent.GetContent())
	case "done":
		return NewToolDoneEvent(toolEvent.GetToolName(), toolCallID, toolEvent.GetMetadata())
	case "error":
		return NewToolErrorEvent(toolEvent.GetToolName(), toolCallID, toolEvent.GetError())
	default:
		return NewToolDataEvent(toolEvent.GetToolName(), toolCallID, toolEvent.GetContent())
	}
}

// StreamChannel converts an external tool stream to LLM stream events
func StreamChannel(ctx context.Context, toolStream <-chan ToolStreamEvent, toolCallID string) <-chan StreamEvent {
	output := make(chan StreamEvent, 10)

	go func() {
		defer close(output)

		for {
			select {
			case toolEvent, ok := <-toolStream:
				if !ok {
					return // Tool stream closed
				}

				llmEvent := ConvertToolStreamToLLMStream(toolEvent, toolCallID)

				select {
				case output <- llmEvent:
				case <-ctx.Done():
					return
				}

				// Stop on completion or error
				if toolEvent.GetStatus() == "done" || toolEvent.GetStatus() == "error" {
					return
				}

			case <-ctx.Done():
				return
			}
		}
	}()

	return output
}

// CreateToolStream creates a simple tool stream from a content string
func CreateToolStream(ctx context.Context, toolName, toolCallID, content string) <-chan StreamEvent {
	output := make(chan StreamEvent, 3)

	go func() {
		defer close(output)

		// Send start event
		output <- NewToolStartEvent(toolName, toolCallID, nil)

		// Send data event
		output <- NewToolDataEvent(toolName, toolCallID, content)

		// Send done event
		output <- NewToolDoneEvent(toolName, toolCallID, nil)
	}()

	return output
}
