package llm

import (
	"context"
	"testing"
	"time"
)

func TestMockStreamChatCompletion(t *testing.T) {
	t.Skip("Integration tests moved to ./test package to avoid import cycles")
}

func TestStreamEventTypes(t *testing.T) {
	delta := NewDeltaEvent(0, &MessageDelta{Content: []MessageContent{NewTextContent("test")}})
	done := NewDoneEvent(0, "stop")
	errEvent := NewErrorEvent(&Error{Message: "test error"})

	if !delta.IsDelta() {
		t.Error("Delta event should be delta")
	}
	if delta.IsDone() || delta.IsError() {
		t.Error("Delta event should not be done or error")
	}

	if !done.IsDone() {
		t.Error("Done event should be done")
	}
	if done.IsDelta() || done.IsError() {
		t.Error("Done event should not be delta or error")
	}

	if !errEvent.IsError() {
		t.Error("Error event should be error")
	}
	if errEvent.IsDelta() || errEvent.IsDone() {
		t.Error("Error event should not be delta or done")
	}
}

func TestStreamEvent_ToolResult(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		event    StreamEvent
		expected string
	}{
		{
			name: "tool_result with success",
			event: StreamEvent{
				Type: "tool_result",
				ToolResult: &ToolResult{
					ToolName:   "test_tool",
					ToolCallID: "call_123",
					Status:     "done",
					Content:    "Hello, World!",
				},
			},
			expected: "tool_result",
		},
		{
			name: "tool_result with progress",
			event: StreamEvent{
				Type: "tool_result",
				ToolResult: &ToolResult{
					ToolName:   "analysis_tool",
					ToolCallID: "call_456",
					Status:     "progress",
					Progress: &ToolProgressInfo{
						Phase:        "analyzing",
						Progress:     45.5,
						Message:      "Processing data...",
						ItemsCurrent: 4500,
						ItemsTotal:   10000,
					},
				},
			},
			expected: "tool_result",
		},
		{
			name: "tool_result with error",
			event: StreamEvent{
				Type: "tool_result",
				ToolResult: &ToolResult{
					ToolName:   "failing_tool",
					ToolCallID: "call_789",
					Status:     "error",
					Error: &ToolExecutionError{
						Code:    "EXECUTION_FAILED",
						Message: "Tool execution failed",
						Type:    "runtime_error",
						Details: map[string]interface{}{"reason": "network timeout"},
					},
				},
			},
			expected: "tool_result",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if tt.event.Type != tt.expected {
				t.Errorf("Expected event type %s, got %s", tt.expected, tt.event.Type)
			}

			// Verify tool result is properly structured
			if tt.event.ToolResult == nil {
				t.Fatal("ToolResult should not be nil")
			}

			// Verify required fields
			if tt.event.ToolResult.ToolName == "" {
				t.Error("ToolResult.ToolName should not be empty")
			}

			if tt.event.ToolResult.Status == "" {
				t.Error("ToolResult.Status should not be empty")
			}

			// Verify status-specific fields
			switch tt.event.ToolResult.Status {
			case "progress":
				if tt.event.ToolResult.Progress == nil {
					t.Error("Progress should not be nil for progress status")
				}
			case "error":
				if tt.event.ToolResult.Error == nil {
					t.Error("Error should not be nil for error status")
				}
			case "done":
				if tt.event.ToolResult.Content == "" && tt.event.ToolResult.Metadata == nil {
					t.Error("Content or Metadata should be present for done status")
				}
			}
		})
	}
}

func TestStreamEvent_Constructors(t *testing.T) {
	t.Parallel()

	t.Run("NewToolProgressEvent", func(t *testing.T) {
		t.Parallel()

		progress := &ToolProgressInfo{
			Phase:        "processing",
			Progress:     25.0,
			Message:      "Processing data...",
			ItemsCurrent: 25,
			ItemsTotal:   100,
		}
		event := NewToolProgressEvent("test_tool", "call_123", progress)

		if event.Type != "tool_result" {
			t.Errorf("Expected event type 'tool_result', got %s", event.Type)
		}

		if event.ToolResult == nil {
			t.Fatal("ToolResult should not be nil")
		}

		if event.ToolResult.ToolName != "test_tool" {
			t.Errorf("Expected tool name 'test_tool', got %s", event.ToolResult.ToolName)
		}

		if event.ToolResult.Status != "progress" {
			t.Errorf("Expected status 'progress', got %s", event.ToolResult.Status)
		}

		if event.ToolResult.Progress == nil {
			t.Fatal("Progress should not be nil")
		}

		if event.ToolResult.Progress.Progress != 25.0 {
			t.Errorf("Expected percentage 25.0, got %f", event.ToolResult.Progress.Progress)
		}
	})

	t.Run("NewToolDataEvent", func(t *testing.T) {
		t.Parallel()

		event := NewToolDataEvent("test_tool", "call_123", "success data")

		if event.Type != "tool_result" {
			t.Errorf("Expected event type 'tool_result', got %s", event.Type)
		}

		if event.ToolResult.Status != "data" {
			t.Errorf("Expected status 'data', got %s", event.ToolResult.Status)
		}

		if event.ToolResult.Content != "success data" {
			t.Errorf("Expected content 'success data', got %s", event.ToolResult.Content)
		}
	})

	t.Run("NewToolErrorEvent", func(t *testing.T) {
		t.Parallel()

		toolError := &ToolExecutionError{
			Code:    "EXEC_FAILED",
			Message: "Execution failed",
			Type:    "runtime_error",
		}
		event := NewToolErrorEvent("test_tool", "call_123", toolError)

		if event.Type != "tool_result" {
			t.Errorf("Expected event type 'tool_result', got %s", event.Type)
		}

		if event.ToolResult.Status != "error" {
			t.Errorf("Expected status 'error', got %s", event.ToolResult.Status)
		}

		if event.ToolResult.Error == nil {
			t.Fatal("Error should not be nil")
		}

		if event.ToolResult.Error.Code != "EXEC_FAILED" {
			t.Errorf("Expected error code 'EXEC_FAILED', got %s", event.ToolResult.Error.Code)
		}
	})
}

func TestStreamMerger(t *testing.T) {
	t.Parallel()

	t.Run("MergeMultipleStreams", func(t *testing.T) {
		t.Parallel()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		// Create multiple input streams
		llmStream := make(chan StreamEvent, 2)
		toolStream := make(chan StreamEvent, 2)

		// Create merger
		merger := NewStreamMerger(ctx, llmStream, []<-chan StreamEvent{toolStream})
		output := merger.Start()

		// Send events to both streams
		go func() {
			defer close(llmStream)
			llmStream <- NewDeltaEvent(0, &MessageDelta{
				Content: []MessageContent{NewTextContent("Hello ")},
			})
			time.Sleep(10 * time.Millisecond)
			llmStream <- NewDeltaEvent(0, &MessageDelta{
				Content: []MessageContent{NewTextContent("from LLM")},
			})
		}()

		go func() {
			defer close(toolStream)
			progress := &ToolProgressInfo{
				Phase:    "processing",
				Progress: 50.0,
				Message:  "Halfway done",
			}
			toolStream <- NewToolProgressEvent("test_tool", "call_123", progress)
			time.Sleep(15 * time.Millisecond)
			toolStream <- NewToolDoneEvent("test_tool", "call_123", map[string]interface{}{"result": "completed"})
		}()

		// Collect all events
		var events []StreamEvent
		for event := range output {
			events = append(events, event)
			if len(events) >= 4 {
				break
			}
		}

		// Verify we received events from both streams
		if len(events) != 4 {
			t.Errorf("Expected 4 events, got %d", len(events))
		}

		// Check that we have both delta and tool_result events
		hasDeltas := false
		hasToolResults := false
		for _, event := range events {
			switch event.Type {
			case "delta":
				hasDeltas = true
			case "tool_result":
				hasToolResults = true
			}
		}

		if !hasDeltas {
			t.Error("Expected to receive delta events")
		}

		if !hasToolResults {
			t.Error("Expected to receive tool_result events")
		}
	})

	t.Run("ContextCancellation", func(t *testing.T) {
		t.Parallel()

		ctx, cancel := context.WithCancel(context.Background())

		llmStream := make(chan StreamEvent, 1)
		merger := NewStreamMerger(ctx, llmStream, nil)
		output := merger.Start()

		// Send one event
		llmStream <- NewDeltaEvent(0, &MessageDelta{
			Content: []MessageContent{NewTextContent("test")},
		})

		// Cancel context
		cancel()

		// Close input stream
		close(llmStream)

		// Verify output stream is closed
		select {
		case event, ok := <-output:
			if ok {
				t.Logf("Received event before closure: %+v", event)
			}
		case <-time.After(100 * time.Millisecond):
			// Expected behavior - output should close
		}

		// Verify channel is eventually closed
		for range output {
			// Drain any remaining events
		}
	})
}

func TestStreamMerger_ErrorHandling(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Create streams with different event types
	errorStream := make(chan StreamEvent, 1)
	normalStream := make(chan StreamEvent, 1)

	merger := NewStreamMerger(ctx, normalStream, []<-chan StreamEvent{errorStream})
	output := merger.Start()

	// Send error event
	go func() {
		defer close(errorStream)
		toolError := &ToolExecutionError{
			Code:    "NETWORK_ERROR",
			Message: "Connection failed",
			Type:    "network_error",
		}
		errorStream <- NewToolErrorEvent("failing_tool", "call_123", toolError)
	}()

	// Send normal event
	go func() {
		defer close(normalStream)
		normalStream <- NewDeltaEvent(0, &MessageDelta{
			Content: []MessageContent{NewTextContent("Normal content")},
		})
	}()

	// Collect events
	var events []StreamEvent
	for event := range output {
		events = append(events, event)
		if len(events) >= 2 {
			break
		}
	}

	if len(events) != 2 {
		t.Errorf("Expected 2 events, got %d", len(events))
	}

	// Verify both error and normal events are received
	hasError := false
	hasNormal := false
	for _, event := range events {
		if event.Type == "tool_result" && event.ToolResult != nil && event.ToolResult.Status == "error" {
			hasError = true
		}
		if event.Type == "delta" {
			hasNormal = true
		}
	}

	if !hasError {
		t.Error("Expected to receive error event")
	}

	if !hasNormal {
		t.Error("Expected to receive normal event")
	}
}

func TestMergeStreams_Utility(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Create streams
	llmStream := make(chan StreamEvent, 1)
	toolStream1 := make(chan StreamEvent, 1)
	toolStream2 := make(chan StreamEvent, 1)

	// Use utility function
	output := MergeStreams(ctx, llmStream, toolStream1, toolStream2)

	// Send events
	go func() {
		defer close(llmStream)
		llmStream <- NewDeltaEvent(0, &MessageDelta{
			Content: []MessageContent{NewTextContent("LLM output")},
		})
	}()

	go func() {
		defer close(toolStream1)
		toolStream1 <- NewToolDataEvent("tool1", "call_1", "Tool 1 data")
	}()

	go func() {
		defer close(toolStream2)
		toolStream2 <- NewToolDataEvent("tool2", "call_2", "Tool 2 data")
	}()

	// Collect events
	var events []StreamEvent
	for event := range output {
		events = append(events, event)
		if len(events) >= 3 {
			break
		}
	}

	if len(events) != 3 {
		t.Errorf("Expected 3 events, got %d", len(events))
	}

	// Verify all stream types are represented
	hasLLM := false
	hasTool1 := false
	hasTool2 := false

	for _, event := range events {
		if event.Type == "delta" {
			hasLLM = true
		}
		if event.Type == "tool_result" && event.ToolResult != nil {
			if event.ToolResult.ToolName == "tool1" {
				hasTool1 = true
			}
			if event.ToolResult.ToolName == "tool2" {
				hasTool2 = true
			}
		}
	}

	if !hasLLM {
		t.Error("Expected LLM event")
	}
	if !hasTool1 {
		t.Error("Expected tool1 event")
	}
	if !hasTool2 {
		t.Error("Expected tool2 event")
	}
}
