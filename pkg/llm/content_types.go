// Multi-modal content types and interface
package llm

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
