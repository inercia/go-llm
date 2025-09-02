// Message types and functionality
package llm

import (
	"encoding/json"
	"fmt"
)

// Message represents a single chat message with multi-modal content support
type Message struct {
	Role       MessageRole      `json:"role"`
	Content    []MessageContent `json:"content"`
	ToolCalls  []ToolCall       `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	Metadata   map[string]any   `json:"metadata,omitempty"`
}

// MessageRole defines the role of a message sender
type MessageRole string

const (
	RoleSystem    MessageRole = "system"
	RoleUser      MessageRole = "user"
	RoleAssistant MessageRole = "assistant"
	RoleTool      MessageRole = "tool"
)

// NewTextMessage creates a new Message with TextContent for backward compatibility
func NewTextMessage(role MessageRole, text string) Message {
	return Message{
		Role:    role,
		Content: []MessageContent{NewTextContent(text)},
	}
}

// GetText extracts text from the first TextContent item for backward compatibility
// Returns empty string if no TextContent is found
func (m Message) GetText() string {
	for _, content := range m.Content {
		if content.Type() == MessageTypeText {
			if textContent, ok := content.(*TextContent); ok {
				return textContent.GetText()
			}
		}
	}
	return ""
}

// SetText sets the message content to a single TextContent item for backward compatibility
// This replaces all existing content with the new text content
func (m *Message) SetText(text string) {
	m.Content = []MessageContent{NewTextContent(text)}
}

// IsTextOnly checks if the message contains only text content
func (m Message) IsTextOnly() bool {
	if len(m.Content) == 0 {
		return false
	}

	for _, content := range m.Content {
		if content.Type() != MessageTypeText {
			return false
		}
	}
	return true
}

// GetContentByType returns all content items of the specified type
func (m Message) GetContentByType(messageType MessageType) []MessageContent {
	var result []MessageContent
	for _, content := range m.Content {
		if content.Type() == messageType {
			result = append(result, content)
		}
	}
	return result
}

// HasContentType checks if the message contains any content of the specified type
func (m Message) HasContentType(messageType MessageType) bool {
	for _, content := range m.Content {
		if content.Type() == messageType {
			return true
		}
	}
	return false
}

// TotalSize returns the sum of all content sizes
func (m Message) TotalSize() int64 {
	var total int64
	for _, content := range m.Content {
		total += content.Size()
	}
	return total
}

// AddContent adds a MessageContent item to the message
func (m *Message) AddContent(content MessageContent) {
	if m.Content == nil {
		m.Content = make([]MessageContent, 0)
	}
	m.Content = append(m.Content, content)
}

// SetMetadata sets a metadata key-value pair
func (m *Message) SetMetadata(key string, value any) {
	if m.Metadata == nil {
		m.Metadata = make(map[string]any)
	}
	m.Metadata[key] = value
}

// GetMetadata retrieves a metadata value by key
func (m Message) GetMetadata(key string) (any, bool) {
	if m.Metadata == nil {
		return nil, false
	}
	value, exists := m.Metadata[key]
	return value, exists
}

// Validate validates all content items in the message
func (m Message) Validate() error {
	for i, content := range m.Content {
		if err := content.Validate(); err != nil {
			return fmt.Errorf("content item %d validation failed: %w", i, err)
		}
	}
	return nil
}

// HasToolCalls checks if the message contains any tool calls
func (m Message) HasToolCalls() bool {
	return len(m.ToolCalls) > 0
}

// GetToolCallByName returns the first tool call with the specified name
func (m Message) GetToolCallByName(name string) (*ToolCall, bool) {
	for _, toolCall := range m.ToolCalls {
		if toolCall.Function.Name == name {
			return &toolCall, true
		}
	}
	return nil, false
}

// GetToolCallsByName returns all tool calls with the specified name
func (m Message) GetToolCallsByName(name string) []ToolCall {
	var result []ToolCall
	for _, toolCall := range m.ToolCalls {
		if toolCall.Function.Name == name {
			result = append(result, toolCall)
		}
	}
	return result
}

// IsToolCallOnly checks if the message contains only tool calls (no text content)
func (m Message) IsToolCallOnly() bool {
	return len(m.ToolCalls) > 0 && (len(m.Content) == 0 || (len(m.Content) == 1 && m.GetText() == ""))
}

// AddToolCall adds a tool call to the message
func (m *Message) AddToolCall(toolCall ToolCall) {
	if m.ToolCalls == nil {
		m.ToolCalls = make([]ToolCall, 0)
	}
	m.ToolCalls = append(m.ToolCalls, toolCall)
}

// DeepCopy creates a deep copy of the message, including all content and tool calls
// This method ensures that modifications to the copy will not affect the original message,
// preventing issues with shared mutable state in concurrent environments
func (m Message) DeepCopy() Message {
	// Create the base copy with simple fields
	copy := Message{
		Role:       m.Role,
		ToolCallID: m.ToolCallID,
	}

	// Deep copy the Content slice
	if len(m.Content) > 0 {
		copy.Content = make([]MessageContent, 0, len(m.Content))
		for _, content := range m.Content {
			copy.Content = append(copy.Content, deepCopyMessageContent(content))
		}
	}

	// Deep copy the ToolCalls slice
	if len(m.ToolCalls) > 0 {
		copy.ToolCalls = make([]ToolCall, 0, len(m.ToolCalls))
		for _, toolCall := range m.ToolCalls {
			copy.ToolCalls = append(copy.ToolCalls, deepCopyToolCall(toolCall))
		}
	}

	// Deep copy the Metadata map
	if len(m.Metadata) > 0 {
		copy.Metadata = make(map[string]any, len(m.Metadata))
		for k, v := range m.Metadata {
			copy.Metadata[k] = deepCopyValue(v)
		}
	}

	return copy
}

// deepCopyMessageContent creates a deep copy of MessageContent based on its type
func deepCopyMessageContent(content MessageContent) MessageContent {
	if content == nil {
		return nil
	}

	switch c := content.(type) {
	case *TextContent:
		return &TextContent{
			Text: c.Text,
		}
	case *ImageContent:
		// Create a new slice to avoid sharing the underlying array
		var dataCopy []byte
		if len(c.Data) > 0 {
			dataCopy = make([]byte, len(c.Data))
			copy(dataCopy, c.Data)
		}
		return &ImageContent{
			Data:     dataCopy,
			URL:      c.URL,
			MimeType: c.MimeType,
		}
	case *FileContent:
		// Create a new slice to avoid sharing the underlying array
		var dataCopy []byte
		if len(c.Data) > 0 {
			dataCopy = make([]byte, len(c.Data))
			copy(dataCopy, c.Data)
		}
		return &FileContent{
			Data:     dataCopy,
			URL:      c.URL,
			MimeType: c.MimeType,
			Filename: c.Filename,
		}
	default:
		// For unknown content types, attempt to use JSON serialization as a fallback
		// This is not the most efficient but ensures compatibility with future content types
		jsonData, err := json.Marshal(content)
		if err != nil {
			// If marshaling fails, return a text content with error information
			return &TextContent{Text: "[DeepCopy Error: Could not copy unknown content type]"}
		}

		// Try to unmarshal back into the same type using reflection-like approach
		// For now, convert to TextContent as a safe fallback
		return &TextContent{Text: string(jsonData)}
	}
}

// deepCopyToolCall creates a deep copy of a ToolCall
func deepCopyToolCall(toolCall ToolCall) ToolCall {
	return ToolCall{
		ID:   toolCall.ID,
		Type: toolCall.Type,
		Function: ToolCallFunction{
			Name:      toolCall.Function.Name,
			Arguments: toolCall.Function.Arguments,
		},
	}
}

// deepCopyValue creates a deep copy of an arbitrary value used in metadata
// This handles common types and uses JSON serialization as a fallback for complex types
func deepCopyValue(v any) any {
	if v == nil {
		return nil
	}

	switch val := v.(type) {
	case string, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64, bool:
		// Primitive types are copied by value
		return val
	case []byte:
		// Deep copy byte slices
		if len(val) == 0 {
			return []byte{}
		}
		copied := make([]byte, len(val))
		copy(copied, val)
		return copied
	case map[string]any:
		// Deep copy string maps
		if len(val) == 0 {
			return make(map[string]any)
		}
		copy := make(map[string]any, len(val))
		for k, v := range val {
			copy[k] = deepCopyValue(v)
		}
		return copy
	case []any:
		// Deep copy slices
		if len(val) == 0 {
			return []any{}
		}
		copy := make([]any, len(val))
		for i, v := range val {
			copy[i] = deepCopyValue(v)
		}
		return copy
	default:
		// For complex types, use JSON serialization as a fallback
		// This is not the most efficient but ensures correctness for most types
		jsonData, err := json.Marshal(val)
		if err != nil {
			// If marshaling fails, return a string representation
			return fmt.Sprintf("[DeepCopy Error: %v]", val)
		}

		var result any
		if err := json.Unmarshal(jsonData, &result); err != nil {
			// If unmarshaling fails, return the JSON string
			return string(jsonData)
		}

		return result
	}
}

// MarshalJSON implements custom JSON marshaling for Message
func (m Message) MarshalJSON() ([]byte, error) {
	type Alias Message

	// Create a temporary struct with content as raw JSON
	temp := struct {
		Alias
		Content []json.RawMessage `json:"content"`
	}{
		Alias: (Alias)(m),
	}

	// Marshal each content item individually
	if len(m.Content) > 0 {
		temp.Content = make([]json.RawMessage, len(m.Content))
		for i, content := range m.Content {
			contentBytes, err := json.Marshal(content)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal content item %d: %w", i, err)
			}
			temp.Content[i] = contentBytes
		}
	}

	return json.Marshal(temp)
}

// UnmarshalJSON implements custom JSON unmarshaling for Message
func (m *Message) UnmarshalJSON(data []byte) error {
	type Alias Message

	// First unmarshal into temporary struct
	temp := struct {
		*Alias
		Content []json.RawMessage `json:"content"`
	}{
		Alias: (*Alias)(m),
	}

	if err := json.Unmarshal(data, &temp); err != nil {
		return err
	}

	// Process content items
	if len(temp.Content) > 0 {
		m.Content = make([]MessageContent, 0, len(temp.Content))

		for i, contentBytes := range temp.Content {
			// First unmarshal to get the type
			var typeChecker struct {
				Type MessageType `json:"type"`
			}

			if err := json.Unmarshal(contentBytes, &typeChecker); err != nil {
				return fmt.Errorf("failed to determine type for content item %d: %w", i, err)
			}

			// Create appropriate content type and unmarshal
			var content MessageContent
			switch typeChecker.Type {
			case MessageTypeText:
				content = &TextContent{}
			case MessageTypeImage:
				content = &ImageContent{}
			case MessageTypeFile:
				content = &FileContent{}
			default:
				return fmt.Errorf("unsupported content type: %s", typeChecker.Type)
			}

			if err := json.Unmarshal(contentBytes, content); err != nil {
				return fmt.Errorf("failed to unmarshal content item %d of type %s: %w", i, typeChecker.Type, err)
			}

			m.Content = append(m.Content, content)
		}
	}

	return nil
}
