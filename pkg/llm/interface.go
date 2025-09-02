// Package llm provides abstractions for Large Language Model clients
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// Client defines the core interface that all LLM clients must implement
type Client interface {
	// ChatCompletion performs a chat completion request
	ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error)

	// StreamChatCompletion performs a streaming chat completion request
	StreamChatCompletion(ctx context.Context, req ChatRequest) (<-chan StreamEvent, error)

	// GetModelInfo returns information about the model being used
	GetModelInfo() ModelInfo

	// Close cleans up any resources used by the client
	Close() error
}

// StreamingClient extends Client with tool stream injection capabilities
type StreamingClient interface {
	Client // Embed existing interface

	// StreamChatCompletionWithTools performs streaming with real-time tool injection
	StreamChatCompletionWithTools(ctx context.Context, req ChatRequest, toolStreams []<-chan StreamEvent) (<-chan StreamEvent, error)
}

// ChatRequest represents a chat completion request (provider-agnostic)
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Tools       []Tool    `json:"tools,omitempty"`
	Temperature *float32  `json:"temperature,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	TopP        *float32  `json:"top_p,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
}

// ChatResponse represents a chat completion response (provider-agnostic)
type ChatResponse struct {
	ID      string   `json:"id"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage,omitempty"`
}

// Message represents a single chat message with multi-modal content support
type Message struct {
	Role       MessageRole      `json:"role"`
	Content    []MessageContent `json:"content"`
	ToolCalls  []ToolCall       `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	Metadata   map[string]any   `json:"metadata,omitempty"`
}

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

// MessageRole defines the role of a message sender
type MessageRole string

const (
	RoleSystem    MessageRole = "system"
	RoleUser      MessageRole = "user"
	RoleAssistant MessageRole = "assistant"
	RoleTool      MessageRole = "tool"
)

// Choice represents a single response choice
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason,omitempty"`
}

// Tool represents a function tool that can be called by the LLM
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction defines the function specification for a tool
type ToolFunction struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  interface{} `json:"parameters"`
}

// ToolCall represents a tool call made by the LLM
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction represents the function call details
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ModelInfo contains information about the model
type ModelInfo struct {
	Name              string `json:"name"`
	Provider          string `json:"provider"`
	MaxTokens         int    `json:"max_tokens"`
	SupportsTools     bool   `json:"supports_tools"`
	SupportsVision    bool   `json:"supports_vision"`
	SupportsFiles     bool   `json:"supports_files"`
	SupportsStreaming bool   `json:"supports_streaming"`
}

// Error represents a standardized LLM error
type Error struct {
	Code       string `json:"code"`
	Message    string `json:"message"`
	Type       string `json:"type"`
	StatusCode int    `json:"status_code,omitempty"`
}

func (e *Error) Error() string {
	return e.Message
}

// Removed ClientFactory interface - use concrete Factory type instead

// MessageContent defines the interface for different types of message content
// This enables multi-modal support for text, images, files, and other content types
type MessageContent interface {
	// Type returns the content type identifier
	Type() MessageType
	// Validate checks if the content is valid and meets requirements
	Validate() error
	// Size returns the content size in bytes for resource management
	Size() int64
}

// MessageType represents the type of message content
type MessageType string

// Supported message content types
const (
	MessageTypeText  MessageType = "text"
	MessageTypeImage MessageType = "image"
	MessageTypeFile  MessageType = "file"
)

// IsValidMessageType checks if the given message type is supported
func IsValidMessageType(msgType MessageType) bool {
	switch msgType {
	case MessageTypeText, MessageTypeImage, MessageTypeFile:
		return true
	default:
		return false
	}
}

// GetSupportedMessageTypes returns all supported message types
func GetSupportedMessageTypes() []MessageType {
	return []MessageType{MessageTypeText, MessageTypeImage, MessageTypeFile}
}

// ClientConfig holds configuration for creating LLM clients
type ClientConfig struct {
	Provider   string            `json:"provider"` // openai, gemini, ollama, anthropic, etc.
	Model      string            `json:"model"`
	APIKey     string            `json:"api_key,omitempty"`
	BaseURL    string            `json:"base_url,omitempty"`
	Timeout    time.Duration     `json:"timeout,omitempty"`
	MaxRetries int               `json:"max_retries,omitempty"`
	Extra      map[string]string `json:"extra,omitempty"` // Provider-specific configs
}
