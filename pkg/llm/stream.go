// Package llm provides abstractions for Large Language Model clients
// streaming.go defines types for streaming chat completions

package llm

// StreamEvent represents a single event in the streaming response
type StreamEvent struct {
	Type       string        `json:"type"` // "delta", "done", "error", "tool_result"
	Choice     *StreamChoice `json:"choice,omitempty"`
	Error      *Error        `json:"error,omitempty"`
	ToolResult *ToolResult   `json:"tool_result,omitempty"`
}

// ToolResult represents tool execution data in streaming responses
type ToolResult struct {
	ToolName   string                 `json:"tool_name"`
	ToolCallID string                 `json:"tool_call_id"`
	Status     string                 `json:"status"` // "start", "progress", "data", "done", "error"
	Content    string                 `json:"content,omitempty"`
	Progress   *ToolProgressInfo      `json:"progress,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	Error      *ToolExecutionError    `json:"error,omitempty"`
}

// ToolProgressInfo represents progress information for long-running tools
type ToolProgressInfo struct {
	Phase        string  `json:"phase"`
	Progress     float64 `json:"progress"`
	Message      string  `json:"message"`
	ItemsTotal   int     `json:"items_total"`
	ItemsCurrent int     `json:"items_current"`
}

// ToolExecutionError represents an error in tool execution
type ToolExecutionError struct {
	Code    string                 `json:"code"`
	Message string                 `json:"message"`
	Type    string                 `json:"type"`
	Details map[string]interface{} `json:"details,omitempty"`
}

// StreamChoice represents a choice in the streaming response
type StreamChoice struct {
	Index        int           `json:"index"`
	Delta        *MessageDelta `json:"delta,omitempty"`
	FinishReason string        `json:"finish_reason,omitempty"`
}

// MessageDelta represents incremental updates to a message
type MessageDelta struct {
	Content   []MessageContent `json:"content,omitempty"`
	ToolCalls []ToolCallDelta  `json:"tool_calls,omitempty"`
}

// ToolCallDelta represents an incremental tool call update
type ToolCallDelta struct {
	Index    int                    `json:"index"`
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Function *ToolCallFunctionDelta `json:"function,omitempty"`
}

// ToolCallFunctionDelta represents incremental function call details
type ToolCallFunctionDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// IsDelta returns true if this is a delta event
func (e StreamEvent) IsDelta() bool {
	return e.Type == "delta" && e.Choice != nil && e.Choice.Delta != nil
}

// IsDone returns true if this is a done event
func (e StreamEvent) IsDone() bool {
	return e.Type == "done" && e.Choice != nil
}

// IsError returns true if this is an error event
func (e StreamEvent) IsError() bool {
	return e.Type == "error" && e.Error != nil
}

// IsToolResult returns true if this is a tool result event
func (e StreamEvent) IsToolResult() bool {
	return e.Type == "tool_result" && e.ToolResult != nil
}

// IsToolStart returns true if this is a tool start event
func (e StreamEvent) IsToolStart() bool {
	return e.IsToolResult() && e.ToolResult.Status == "start"
}

// IsToolProgress returns true if this is a tool progress event
func (e StreamEvent) IsToolProgress() bool {
	return e.IsToolResult() && e.ToolResult.Status == "progress"
}

// IsToolData returns true if this is a tool data event
func (e StreamEvent) IsToolData() bool {
	return e.IsToolResult() && e.ToolResult.Status == "data"
}

// IsToolDone returns true if this is a tool done event
func (e StreamEvent) IsToolDone() bool {
	return e.IsToolResult() && e.ToolResult.Status == "done"
}

// IsToolError returns true if this is a tool error event
func (e StreamEvent) IsToolError() bool {
	return e.IsToolResult() && e.ToolResult.Status == "error"
}

// NewDeltaEvent creates a new delta stream event
func NewDeltaEvent(index int, delta *MessageDelta) StreamEvent {
	return StreamEvent{
		Type: "delta",
		Choice: &StreamChoice{
			Index: index,
			Delta: delta,
		},
	}
}

// NewDoneEvent creates a new done stream event
func NewDoneEvent(index int, finishReason string) StreamEvent {
	return StreamEvent{
		Type: "done",
		Choice: &StreamChoice{
			Index:        index,
			FinishReason: finishReason,
		},
	}
}

// NewErrorEvent creates a new error stream event
func NewErrorEvent(err *Error) StreamEvent {
	return StreamEvent{
		Type:  "error",
		Error: err,
	}
}

// NewToolResultEvent creates a new tool result stream event
func NewToolResultEvent(toolResult *ToolResult) StreamEvent {
	return StreamEvent{
		Type:       "tool_result",
		ToolResult: toolResult,
	}
}

// NewToolStartEvent creates a new tool start event
func NewToolStartEvent(toolName, toolCallID string, metadata map[string]interface{}) StreamEvent {
	return NewToolResultEvent(&ToolResult{
		ToolName:   toolName,
		ToolCallID: toolCallID,
		Status:     "start",
		Metadata:   metadata,
	})
}

// NewToolProgressEvent creates a new tool progress event
func NewToolProgressEvent(toolName, toolCallID string, progress *ToolProgressInfo) StreamEvent {
	return NewToolResultEvent(&ToolResult{
		ToolName:   toolName,
		ToolCallID: toolCallID,
		Status:     "progress",
		Progress:   progress,
	})
}

// NewToolDataEvent creates a new tool data event
func NewToolDataEvent(toolName, toolCallID, content string) StreamEvent {
	return NewToolResultEvent(&ToolResult{
		ToolName:   toolName,
		ToolCallID: toolCallID,
		Status:     "data",
		Content:    content,
	})
}

// NewToolDoneEvent creates a new tool done event
func NewToolDoneEvent(toolName, toolCallID string, metadata map[string]interface{}) StreamEvent {
	return NewToolResultEvent(&ToolResult{
		ToolName:   toolName,
		ToolCallID: toolCallID,
		Status:     "done",
		Metadata:   metadata,
	})
}

// NewToolErrorEvent creates a new tool error event
func NewToolErrorEvent(toolName, toolCallID string, toolError *ToolExecutionError) StreamEvent {
	return NewToolResultEvent(&ToolResult{
		ToolName:   toolName,
		ToolCallID: toolCallID,
		Status:     "error",
		Error:      toolError,
	})
}
