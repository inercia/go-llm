package llm

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// Mock implementation of MessageContent for testing
type mockMessageContent struct {
	contentType MessageType
	size        int64
	valid       bool
	validateErr error
}

func (m *mockMessageContent) Type() MessageType {
	return m.contentType
}

func (m *mockMessageContent) Validate() error {
	if !m.valid {
		if m.validateErr != nil {
			return m.validateErr
		}
		return errors.New("mock validation failed")
	}
	return nil
}

func (m *mockMessageContent) Size() int64 {
	return m.size
}

// Test MessageContent interface contract compliance
func TestMessageContentInterface(t *testing.T) {
	tests := []struct {
		name           string
		content        MessageContent
		expectedType   MessageType
		expectedSize   int64
		shouldValidate bool
		expectedError  string
	}{
		{
			name: "valid text content",
			content: &mockMessageContent{
				contentType: MessageTypeText,
				size:        100,
				valid:       true,
			},
			expectedType:   MessageTypeText,
			expectedSize:   100,
			shouldValidate: true,
		},
		{
			name: "valid image content",
			content: &mockMessageContent{
				contentType: MessageTypeImage,
				size:        1024,
				valid:       true,
			},
			expectedType:   MessageTypeImage,
			expectedSize:   1024,
			shouldValidate: true,
		},
		{
			name: "valid file content",
			content: &mockMessageContent{
				contentType: MessageTypeFile,
				size:        2048,
				valid:       true,
			},
			expectedType:   MessageTypeFile,
			expectedSize:   2048,
			shouldValidate: true,
		},
		{
			name: "invalid content",
			content: &mockMessageContent{
				contentType: MessageTypeText,
				size:        50,
				valid:       false,
			},
			expectedType:   MessageTypeText,
			expectedSize:   50,
			shouldValidate: false,
			expectedError:  "mock validation failed",
		},
		{
			name: "invalid content with custom error",
			content: &mockMessageContent{
				contentType: MessageTypeImage,
				size:        0,
				valid:       false,
				validateErr: errors.New("custom validation error"),
			},
			expectedType:   MessageTypeImage,
			expectedSize:   0,
			shouldValidate: false,
			expectedError:  "custom validation error",
		},
		{
			name: "zero size content",
			content: &mockMessageContent{
				contentType: MessageTypeFile,
				size:        0,
				valid:       true,
			},
			expectedType:   MessageTypeFile,
			expectedSize:   0,
			shouldValidate: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test Type() method
			if got := tt.content.Type(); got != tt.expectedType {
				t.Errorf("MessageContent.Type() = %v, want %v", got, tt.expectedType)
			}

			// Test Size() method
			if got := tt.content.Size(); got != tt.expectedSize {
				t.Errorf("MessageContent.Size() = %v, want %v", got, tt.expectedSize)
			}

			// Test Validate() method
			err := tt.content.Validate()
			if tt.shouldValidate {
				if err != nil {
					t.Errorf("MessageContent.Validate() returned error for valid content: %v", err)
				}
			} else {
				if err == nil {
					t.Error("MessageContent.Validate() should have returned error for invalid content")
				} else if tt.expectedError != "" && err.Error() != tt.expectedError {
					t.Errorf("MessageContent.Validate() error = %v, want %v", err.Error(), tt.expectedError)
				}
			}
		})
	}
}

// Test MessageType constants
func TestMessageTypeConstants(t *testing.T) {
	tests := []struct {
		name     string
		msgType  MessageType
		expected string
	}{
		{"text type", MessageTypeText, "text"},
		{"image type", MessageTypeImage, "image"},
		{"file type", MessageTypeFile, "file"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if string(tt.msgType) != tt.expected {
				t.Errorf("MessageType constant = %v, want %v", string(tt.msgType), tt.expected)
			}
		})
	}
}

// Test IsValidMessageType function
func TestIsValidMessageType(t *testing.T) {
	tests := []struct {
		name     string
		msgType  MessageType
		expected bool
	}{
		{"valid text type", MessageTypeText, true},
		{"valid image type", MessageTypeImage, true},
		{"valid file type", MessageTypeFile, true},
		{"invalid empty type", MessageType(""), false},
		{"invalid unknown type", MessageType("unknown"), false},
		{"invalid random type", MessageType("random"), false},
		{"invalid with spaces", MessageType("text "), false},
		{"invalid with uppercase", MessageType("TEXT"), false},
		{"invalid mixed case", MessageType("Text"), false},
		{"invalid special characters", MessageType("text!"), false},
		{"invalid number type", MessageType("123"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsValidMessageType(tt.msgType); got != tt.expected {
				t.Errorf("IsValidMessageType(%v) = %v, want %v", tt.msgType, got, tt.expected)
			}
		})
	}
}

// Test GetSupportedMessageTypes function
func TestGetSupportedMessageTypes(t *testing.T) {
	supported := GetSupportedMessageTypes()

	// Check that we get the expected types
	expectedTypes := []MessageType{MessageTypeText, MessageTypeImage, MessageTypeFile}

	if len(supported) != len(expectedTypes) {
		t.Errorf("GetSupportedMessageTypes() returned %d types, want %d", len(supported), len(expectedTypes))
	}

	// Check that all expected types are present
	typeMap := make(map[MessageType]bool)
	for _, msgType := range supported {
		typeMap[msgType] = true
	}

	for _, expectedType := range expectedTypes {
		if !typeMap[expectedType] {
			t.Errorf("GetSupportedMessageTypes() missing expected type: %v", expectedType)
		}
	}

	// Check that all returned types are valid
	for _, msgType := range supported {
		if !IsValidMessageType(msgType) {
			t.Errorf("GetSupportedMessageTypes() returned invalid type: %v", msgType)
		}
	}

	// Test that the function returns a new slice each time (not the same reference)
	supported2 := GetSupportedMessageTypes()
	if &supported[0] == &supported2[0] {
		t.Error("GetSupportedMessageTypes() should return a new slice, not the same reference")
	}

	// Modify the returned slice to ensure it doesn't affect future calls
	supported[0] = MessageType("modified")
	supported3 := GetSupportedMessageTypes()
	if supported3[0] != MessageTypeText {
		t.Error("GetSupportedMessageTypes() should be unaffected by modifications to returned slice")
	}
}

// Test edge cases and boundary conditions
func TestMessageTypeEdgeCases(t *testing.T) {
	t.Run("message type string conversion", func(t *testing.T) {
		msgType := MessageTypeText
		str := string(msgType)
		if str != "text" {
			t.Errorf("MessageType string conversion = %v, want 'text'", str)
		}
	})

	t.Run("message type comparison", func(t *testing.T) {
		type1 := MessageTypeText
		type2 := MessageTypeText
		type3 := MessageTypeImage

		if type1 != type2 {
			t.Error("Same MessageType values should be equal")
		}

		if type1 == type3 {
			t.Error("Different MessageType values should not be equal")
		}
	})

	t.Run("message type in map", func(t *testing.T) {
		typeMap := map[MessageType]string{
			MessageTypeText:  "text content",
			MessageTypeImage: "image content",
			MessageTypeFile:  "file content",
		}

		if typeMap[MessageTypeText] != "text content" {
			t.Error("MessageType should work as map key")
		}
	})

	t.Run("message type in switch", func(t *testing.T) {
		testType := MessageTypeImage
		var result string

		switch testType {
		case MessageTypeText:
			result = "text"
		case MessageTypeImage:
			result = "image"
		case MessageTypeFile:
			result = "file"
		default:
			result = "unknown"
		}

		if result != "image" {
			t.Errorf("MessageType in switch statement = %v, want 'image'", result)
		}
	})
}

// Test interface nil safety
func TestMessageContentNilSafety(t *testing.T) {
	var content MessageContent

	// These should not panic, though they will return zero values
	t.Run("nil interface methods", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic when calling methods on nil interface")
			}
		}()

		_ = content.Type()
	})

	// Test with nil pointer to struct that implements interface
	var mockContent *mockMessageContent
	var iface MessageContent = mockContent

	t.Run("nil pointer interface methods", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic when calling methods on nil pointer implementing interface")
			}
		}()

		_ = iface.Type()
	})
}

// Benchmark tests for performance validation
func BenchmarkIsValidMessageType(b *testing.B) {
	testTypes := []MessageType{
		MessageTypeText,
		MessageTypeImage,
		MessageTypeFile,
		MessageType("invalid"),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, msgType := range testTypes {
			IsValidMessageType(msgType)
		}
	}
}

func BenchmarkGetSupportedMessageTypes(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GetSupportedMessageTypes()
	}
}

func BenchmarkMessageContentInterface(b *testing.B) {
	content := &mockMessageContent{
		contentType: MessageTypeText,
		size:        1000,
		valid:       true,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = content.Type()
		_ = content.Size()
		_ = content.Validate()
	}
}

// Test Message struct with multi-modal content support
func TestNewTextMessage(t *testing.T) {
	tests := []struct {
		name string
		role MessageRole
		text string
	}{
		{
			name: "system message",
			role: RoleSystem,
			text: "You are a helpful assistant",
		},
		{
			name: "user message",
			role: RoleUser,
			text: "Hello, how are you?",
		},
		{
			name: "assistant message",
			role: RoleAssistant,
			text: "I'm doing well, thank you!",
		},
		{
			name: "empty text",
			role: RoleUser,
			text: "",
		},
		{
			name: "unicode text",
			role: RoleUser,
			text: "Hello 世界! 🌍",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := NewTextMessage(tt.role, tt.text)

			if msg.Role != tt.role {
				t.Errorf("NewTextMessage() role = %v, want %v", msg.Role, tt.role)
			}

			if len(msg.Content) != 1 {
				t.Errorf("NewTextMessage() content length = %d, want 1", len(msg.Content))
				return
			}

			if msg.Content[0].Type() != MessageTypeText {
				t.Errorf("NewTextMessage() content type = %v, want %v", msg.Content[0].Type(), MessageTypeText)
			}

			if msg.GetText() != tt.text {
				t.Errorf("NewTextMessage() GetText() = %v, want %v", msg.GetText(), tt.text)
			}

			if !msg.IsTextOnly() {
				t.Error("NewTextMessage() should create text-only message")
			}

			if msg.Metadata != nil {
				t.Error("NewTextMessage() should not initialize Metadata")
			}
		})
	}
}

func TestMessageGetText(t *testing.T) {
	tests := []struct {
		name     string
		message  Message
		expected string
	}{
		{
			name:     "empty message",
			message:  Message{},
			expected: "",
		},
		{
			name: "text only message",
			message: Message{
				Content: []MessageContent{
					NewTextContent("Hello world"),
				},
			},
			expected: "Hello world",
		},
		{
			name: "mixed content - text first",
			message: Message{
				Content: []MessageContent{
					NewTextContent("First text"),
					NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
					NewTextContent("Second text"),
				},
			},
			expected: "First text",
		},
		{
			name: "mixed content - text second",
			message: Message{
				Content: []MessageContent{
					NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
					NewTextContent("Found text"),
				},
			},
			expected: "Found text",
		},
		{
			name: "no text content",
			message: Message{
				Content: []MessageContent{
					NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
				},
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.message.GetText(); got != tt.expected {
				t.Errorf("Message.GetText() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestMessageIsTextOnly(t *testing.T) {
	tests := []struct {
		name     string
		message  Message
		expected bool
	}{
		{
			name:     "empty message",
			message:  Message{},
			expected: false,
		},
		{
			name: "single text content",
			message: Message{
				Content: []MessageContent{
					NewTextContent("Hello"),
				},
			},
			expected: true,
		},
		{
			name: "multiple text content",
			message: Message{
				Content: []MessageContent{
					NewTextContent("Hello"),
					NewTextContent("World"),
				},
			},
			expected: true,
		},
		{
			name: "mixed content",
			message: Message{
				Content: []MessageContent{
					NewTextContent("Hello"),
					NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
				},
			},
			expected: false,
		},
		{
			name: "image only",
			message: Message{
				Content: []MessageContent{
					NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
				},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.message.IsTextOnly(); got != tt.expected {
				t.Errorf("Message.IsTextOnly() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestMessageGetContentByType(t *testing.T) {
	// Create sample content
	textContent1 := NewTextContent("Hello")
	textContent2 := NewTextContent("World")
	imageContent := NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg")
	fileContent := NewFileContentFromURL("https://example.com/file.pdf", "document.pdf", "application/pdf", 1024)

	message := Message{
		Content: []MessageContent{
			textContent1,
			imageContent,
			textContent2,
			fileContent,
		},
	}

	tests := []struct {
		name          string
		messageType   MessageType
		expectedCount int
	}{
		{
			name:          "get text content",
			messageType:   MessageTypeText,
			expectedCount: 2,
		},
		{
			name:          "get image content",
			messageType:   MessageTypeImage,
			expectedCount: 1,
		},
		{
			name:          "get file content",
			messageType:   MessageTypeFile,
			expectedCount: 1,
		},
		{
			name:          "get nonexistent type",
			messageType:   MessageType("nonexistent"),
			expectedCount: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := message.GetContentByType(tt.messageType)
			if len(got) != tt.expectedCount {
				t.Errorf("Message.GetContentByType(%v) returned %d items, want %d", tt.messageType, len(got), tt.expectedCount)
			}

			// Verify all returned content is of the correct type
			for i, content := range got {
				if content.Type() != tt.messageType {
					t.Errorf("Message.GetContentByType(%v)[%d] has type %v, want %v", tt.messageType, i, content.Type(), tt.messageType)
				}
			}
		})
	}
}

func TestMessageHasContentType(t *testing.T) {
	message := Message{
		Content: []MessageContent{
			NewTextContent("Hello"),
			NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
		},
	}

	tests := []struct {
		name        string
		messageType MessageType
		expected    bool
	}{
		{
			name:        "has text",
			messageType: MessageTypeText,
			expected:    true,
		},
		{
			name:        "has image",
			messageType: MessageTypeImage,
			expected:    true,
		},
		{
			name:        "no file",
			messageType: MessageTypeFile,
			expected:    false,
		},
		{
			name:        "nonexistent type",
			messageType: MessageType("nonexistent"),
			expected:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := message.HasContentType(tt.messageType); got != tt.expected {
				t.Errorf("Message.HasContentType(%v) = %v, want %v", tt.messageType, got, tt.expected)
			}
		})
	}

	// Test empty message
	emptyMessage := Message{}
	if emptyMessage.HasContentType(MessageTypeText) {
		t.Error("Empty message should not have any content type")
	}
}

func TestMessageTotalSize(t *testing.T) {
	tests := []struct {
		name     string
		message  Message
		expected int64
	}{
		{
			name:     "empty message",
			message:  Message{},
			expected: 0,
		},
		{
			name: "single content",
			message: Message{
				Content: []MessageContent{
					NewTextContent("Hello"), // 5 bytes
				},
			},
			expected: 5,
		},
		{
			name: "multiple content",
			message: Message{
				Content: []MessageContent{
					NewTextContent("Hello"), // 5 bytes
					NewTextContent("World"), // 5 bytes
					NewFileContentFromURL("https://example.com/file.pdf", "document.pdf", "application/pdf", 1024), // 1024 bytes
				},
			},
			expected: 1034,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.message.TotalSize(); got != tt.expected {
				t.Errorf("Message.TotalSize() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestMessageAddContent(t *testing.T) {
	msg := &Message{
		Role: RoleUser,
	}

	// Test adding to nil content slice
	textContent := NewTextContent("Hello")
	msg.AddContent(textContent)

	if len(msg.Content) != 1 {
		t.Errorf("AddContent() resulted in %d content items, want 1", len(msg.Content))
	}

	if msg.Content[0] != textContent {
		t.Error("AddContent() did not add the correct content")
	}

	// Test adding to existing content
	imageContent := NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg")
	msg.AddContent(imageContent)

	if len(msg.Content) != 2 {
		t.Errorf("AddContent() resulted in %d content items, want 2", len(msg.Content))
	}

	if msg.Content[1] != imageContent {
		t.Error("AddContent() did not add the second content correctly")
	}
}

func TestMessageMetadata(t *testing.T) {
	msg := &Message{
		Role: RoleUser,
	}

	// Test setting metadata on nil map
	msg.SetMetadata("key1", "value1")

	if msg.Metadata == nil {
		t.Error("SetMetadata() did not initialize Metadata map")
	}

	if value, exists := msg.GetMetadata("key1"); !exists || value != "value1" {
		t.Errorf("GetMetadata('key1') = %v, %v, want 'value1', true", value, exists)
	}

	// Test setting multiple metadata items
	msg.SetMetadata("key2", 42)
	msg.SetMetadata("key3", true)

	if len(msg.Metadata) != 3 {
		t.Errorf("Metadata map has %d items, want 3", len(msg.Metadata))
	}

	// Test getting nonexistent key
	if value, exists := msg.GetMetadata("nonexistent"); exists {
		t.Errorf("GetMetadata('nonexistent') = %v, %v, want nil, false", value, exists)
	}

	// Test getting from nil metadata
	emptyMsg := Message{}
	if value, exists := emptyMsg.GetMetadata("key1"); exists {
		t.Errorf("GetMetadata on empty message = %v, %v, want nil, false", value, exists)
	}
}

func TestMessageValidate(t *testing.T) {
	tests := []struct {
		name      string
		message   Message
		wantError bool
		errorMsg  string
	}{
		{
			name: "valid message",
			message: Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Hello"),
					NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
				},
			},
			wantError: false,
		},
		{
			name: "empty message",
			message: Message{
				Role: RoleUser,
			},
			wantError: false,
		},
		{
			name: "message with invalid content",
			message: Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Valid text"),
					&mockMessageContent{
						contentType: MessageTypeText,
						valid:       false,
					},
				},
			},
			wantError: true,
			errorMsg:  "content item 1 validation failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.message.Validate()
			if tt.wantError {
				if err == nil {
					t.Error("Message.Validate() expected error but got none")
				} else if tt.errorMsg != "" && !strings.Contains(err.Error(), tt.errorMsg) {
					t.Errorf("Message.Validate() error = %v, want error containing %v", err, tt.errorMsg)
				}
			} else {
				if err != nil {
					t.Errorf("Message.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

func TestMessageJSONMarshalUnmarshal(t *testing.T) {
	original := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("Hello world"),
			NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
			NewFileContentFromBytes([]byte("file content"), "test.txt", "text/plain"),
		},
		ToolCalls: []ToolCall{
			{
				ID:   "call_1",
				Type: "function",
				Function: ToolCallFunction{
					Name:      "test_function",
					Arguments: `{"arg": "value"}`,
				},
			},
		},
		ToolCallID: "tool_1",
		Metadata: map[string]any{
			"source":   "test",
			"priority": 1.0,
		},
	}

	// Marshal to JSON
	jsonData, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("Failed to marshal message: %v", err)
	}

	// Unmarshal from JSON
	var restored Message
	err = json.Unmarshal(jsonData, &restored)
	if err != nil {
		t.Fatalf("Failed to unmarshal message: %v", err)
	}

	// Verify basic fields
	if restored.Role != original.Role {
		t.Errorf("Role mismatch: got %v, want %v", restored.Role, original.Role)
	}

	if restored.ToolCallID != original.ToolCallID {
		t.Errorf("ToolCallID mismatch: got %v, want %v", restored.ToolCallID, original.ToolCallID)
	}

	// Verify content
	if len(restored.Content) != len(original.Content) {
		t.Errorf("Content length mismatch: got %d, want %d", len(restored.Content), len(original.Content))
	}

	// Verify content types
	for i, content := range restored.Content {
		expectedType := original.Content[i].Type()
		if content.Type() != expectedType {
			t.Errorf("Content[%d] type mismatch: got %v, want %v", i, content.Type(), expectedType)
		}
	}

	// Verify text content
	if restored.GetText() != original.GetText() {
		t.Errorf("Text content mismatch: got %v, want %v", restored.GetText(), original.GetText())
	}

	// Verify metadata
	if len(restored.Metadata) != len(original.Metadata) {
		t.Errorf("Metadata length mismatch: got %d, want %d", len(restored.Metadata), len(original.Metadata))
	}

	for key, expectedValue := range original.Metadata {
		if actualValue, exists := restored.Metadata[key]; !exists {
			t.Errorf("Metadata key %v missing", key)
		} else if actualValue != expectedValue {
			t.Errorf("Metadata[%v] mismatch: got %v, want %v", key, actualValue, expectedValue)
		}
	}
}

func TestMessageJSONEdgeCases(t *testing.T) {
	t.Run("empty message", func(t *testing.T) {
		original := Message{Role: RoleUser}

		data, err := json.Marshal(original)
		if err != nil {
			t.Fatalf("Marshal error: %v", err)
		}

		var restored Message
		err = json.Unmarshal(data, &restored)
		if err != nil {
			t.Fatalf("Unmarshal error: %v", err)
		}

		if restored.Role != original.Role {
			t.Errorf("Role mismatch: got %v, want %v", restored.Role, original.Role)
		}

		if len(restored.Content) != 0 {
			t.Errorf("Expected empty content, got %d items", len(restored.Content))
		}
	})

	t.Run("unsupported content type", func(t *testing.T) {
		// This test simulates receiving JSON with an unsupported content type
		invalidJSON := `{
			"role": "user",
			"content": [
				{"type": "unsupported", "data": "test"}
			]
		}`

		var msg Message
		err := json.Unmarshal([]byte(invalidJSON), &msg)
		if err == nil {
			t.Error("Expected error for unsupported content type")
		}
	})

	t.Run("malformed content JSON", func(t *testing.T) {
		invalidJSON := `{
			"role": "user",
			"content": [
				{"type": "text", "invalid": }
			]
		}`

		var msg Message
		err := json.Unmarshal([]byte(invalidJSON), &msg)
		if err == nil {
			t.Error("Expected error for malformed JSON")
		}
	})
}

// Test backward compatibility with existing workflows
func TestMessageBackwardCompatibility(t *testing.T) {
	// Simulate old-style message creation patterns
	messages := []Message{
		NewTextMessage(RoleSystem, "You are a helpful assistant"),
		NewTextMessage(RoleUser, "Hello!"),
		NewTextMessage(RoleAssistant, "Hi there! How can I help you today?"),
	}

	// Test that GetText() works as expected
	expectedTexts := []string{
		"You are a helpful assistant",
		"Hello!",
		"Hi there! How can I help you today?",
	}

	for i, msg := range messages {
		if got := msg.GetText(); got != expectedTexts[i] {
			t.Errorf("Message[%d].GetText() = %v, want %v", i, got, expectedTexts[i])
		}

		if !msg.IsTextOnly() {
			t.Errorf("Message[%d] should be text-only", i)
		}
	}

	// Test JSON serialization maintains compatibility
	for i, msg := range messages {
		data, err := json.Marshal(msg)
		if err != nil {
			t.Errorf("Failed to marshal message[%d]: %v", i, err)
			continue
		}

		var restored Message
		err = json.Unmarshal(data, &restored)
		if err != nil {
			t.Errorf("Failed to unmarshal message[%d]: %v", i, err)
			continue
		}

		if restored.GetText() != expectedTexts[i] {
			t.Errorf("Restored message[%d].GetText() = %v, want %v", i, restored.GetText(), expectedTexts[i])
		}
	}
}

func TestMessage_HasToolCalls(t *testing.T) {
	tests := []struct {
		name     string
		message  Message
		expected bool
	}{
		{
			name: "message_with_tool_calls",
			message: Message{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: ToolCallFunction{
							Name:      "search",
							Arguments: `{"query": "test"}`,
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "message_without_tool_calls",
			message: Message{
				Role:    RoleAssistant,
				Content: []MessageContent{NewTextContent("Hello")},
			},
			expected: false,
		},
		{
			name: "message_with_empty_tool_calls",
			message: Message{
				Role:      RoleAssistant,
				ToolCalls: []ToolCall{},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := tt.message.HasToolCalls()
			if result != tt.expected {
				t.Errorf("HasToolCalls() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestMessage_GetToolCallByName(t *testing.T) {
	tests := []struct {
		name       string
		message    Message
		searchName string
		expected   *ToolCall
		found      bool
	}{
		{
			name: "find_existing_tool_call",
			message: Message{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: ToolCallFunction{
							Name:      "search",
							Arguments: `{"query": "test"}`,
						},
					},
					{
						ID:   "call_2",
						Type: "function",
						Function: ToolCallFunction{
							Name:      "calculate",
							Arguments: `{"expression": "2+2"}`,
						},
					},
				},
			},
			searchName: "search",
			expected: &ToolCall{
				ID:   "call_1",
				Type: "function",
				Function: ToolCallFunction{
					Name:      "search",
					Arguments: `{"query": "test"}`,
				},
			},
			found: true,
		},
		{
			name: "tool_call_not_found",
			message: Message{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: ToolCallFunction{
							Name:      "search",
							Arguments: `{"query": "test"}`,
						},
					},
				},
			},
			searchName: "nonexistent",
			expected:   nil,
			found:      false,
		},
		{
			name:       "no_tool_calls",
			message:    Message{Role: RoleAssistant},
			searchName: "search",
			expected:   nil,
			found:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result, found := tt.message.GetToolCallByName(tt.searchName)

			if found != tt.found {
				t.Errorf("GetToolCallByName() found = %v, expected %v", found, tt.found)
			}

			if tt.expected == nil && result != nil {
				t.Errorf("GetToolCallByName() = %v, expected nil", result)
			} else if tt.expected != nil && result == nil {
				t.Errorf("GetToolCallByName() = nil, expected %v", tt.expected)
			} else if tt.expected != nil && result != nil {
				if result.ID != tt.expected.ID || result.Function.Name != tt.expected.Function.Name {
					t.Errorf("GetToolCallByName() = %v, expected %v", result, tt.expected)
				}
			}
		})
	}
}

func TestMessage_GetToolCallsByName(t *testing.T) {
	tests := []struct {
		name       string
		message    Message
		searchName string
		expected   []ToolCall
	}{
		{
			name: "find_multiple_tool_calls",
			message: Message{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: ToolCallFunction{
							Name:      "search",
							Arguments: `{"query": "test1"}`,
						},
					},
					{
						ID:   "call_2",
						Type: "function",
						Function: ToolCallFunction{
							Name:      "calculate",
							Arguments: `{"expression": "2+2"}`,
						},
					},
					{
						ID:   "call_3",
						Type: "function",
						Function: ToolCallFunction{
							Name:      "search",
							Arguments: `{"query": "test2"}`,
						},
					},
				},
			},
			searchName: "search",
			expected: []ToolCall{
				{
					ID:   "call_1",
					Type: "function",
					Function: ToolCallFunction{
						Name:      "search",
						Arguments: `{"query": "test1"}`,
					},
				},
				{
					ID:   "call_3",
					Type: "function",
					Function: ToolCallFunction{
						Name:      "search",
						Arguments: `{"query": "test2"}`,
					},
				},
			},
		},
		{
			name: "no_matching_tool_calls",
			message: Message{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: ToolCallFunction{
							Name:      "search",
							Arguments: `{"query": "test"}`,
						},
					},
				},
			},
			searchName: "nonexistent",
			expected:   []ToolCall{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := tt.message.GetToolCallsByName(tt.searchName)

			if len(result) != len(tt.expected) {
				t.Errorf("GetToolCallsByName() returned %d tool calls, expected %d", len(result), len(tt.expected))
				return
			}

			for i, toolCall := range result {
				if toolCall.ID != tt.expected[i].ID || toolCall.Function.Name != tt.expected[i].Function.Name {
					t.Errorf("GetToolCallsByName()[%d] = %v, expected %v", i, toolCall, tt.expected[i])
				}
			}
		})
	}
}

func TestMessage_IsToolCallOnly(t *testing.T) {
	tests := []struct {
		name     string
		message  Message
		expected bool
	}{
		{
			name: "tool_calls_with_no_content",
			message: Message{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: ToolCallFunction{
							Name:      "search",
							Arguments: `{"query": "test"}`,
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "tool_calls_with_empty_text",
			message: Message{
				Role:    RoleAssistant,
				Content: []MessageContent{NewTextContent("")},
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: ToolCallFunction{
							Name:      "search",
							Arguments: `{"query": "test"}`,
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "tool_calls_with_text_content",
			message: Message{
				Role:    RoleAssistant,
				Content: []MessageContent{NewTextContent("I'll search for that")},
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: ToolCallFunction{
							Name:      "search",
							Arguments: `{"query": "test"}`,
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "no_tool_calls_with_content",
			message: Message{
				Role:    RoleAssistant,
				Content: []MessageContent{NewTextContent("Hello")},
			},
			expected: false,
		},
		{
			name: "no_tool_calls_no_content",
			message: Message{
				Role: RoleAssistant,
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := tt.message.IsToolCallOnly()
			if result != tt.expected {
				t.Errorf("IsToolCallOnly() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestMessage_AddToolCall(t *testing.T) {
	tests := []struct {
		name        string
		message     Message
		toolCall    ToolCall
		expectedLen int
	}{
		{
			name: "add_to_empty_tool_calls",
			message: Message{
				Role: RoleAssistant,
			},
			toolCall: ToolCall{
				ID:   "call_1",
				Type: "function",
				Function: ToolCallFunction{
					Name:      "search",
					Arguments: `{"query": "test"}`,
				},
			},
			expectedLen: 1,
		},
		{
			name: "add_to_existing_tool_calls",
			message: Message{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: ToolCallFunction{
							Name:      "existing",
							Arguments: `{}`,
						},
					},
				},
			},
			toolCall: ToolCall{
				ID:   "call_2",
				Type: "function",
				Function: ToolCallFunction{
					Name:      "search",
					Arguments: `{"query": "test"}`,
				},
			},
			expectedLen: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			tt.message.AddToolCall(tt.toolCall)

			if len(tt.message.ToolCalls) != tt.expectedLen {
				t.Errorf("AddToolCall() resulted in %d tool calls, expected %d", len(tt.message.ToolCalls), tt.expectedLen)
			}

			// Check if the added tool call is present
			found := false
			for _, tc := range tt.message.ToolCalls {
				if tc.ID == tt.toolCall.ID && tc.Function.Name == tt.toolCall.Function.Name {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("AddToolCall() did not add the expected tool call")
			}
		})
	}
}

func TestChoice_WantsToolExecution(t *testing.T) {
	tests := []struct {
		name     string
		choice   Choice
		expected bool
	}{
		{
			name: "finish_reason_tool_calls",
			choice: Choice{
				FinishReason: FinishReasonToolCalls,
				Message: Message{
					Role: RoleAssistant,
				},
			},
			expected: true,
		},
		{
			name: "has_tool_calls_no_finish_reason",
			choice: Choice{
				FinishReason: "",
				Message: Message{
					Role: RoleAssistant,
					ToolCalls: []ToolCall{
						{
							ID:   "call_1",
							Type: "function",
							Function: ToolCallFunction{
								Name:      "search",
								Arguments: `{"query": "test"}`,
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "finish_reason_stop",
			choice: Choice{
				FinishReason: FinishReasonStop,
				Message: Message{
					Role: RoleAssistant,
				},
			},
			expected: false,
		},
		{
			name: "no_tool_calls_no_finish_reason",
			choice: Choice{
				Message: Message{
					Role: RoleAssistant,
				},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := tt.choice.WantsToolExecution()
			if result != tt.expected {
				t.Errorf("WantsToolExecution() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestChoice_IsComplete(t *testing.T) {
	tests := []struct {
		name     string
		choice   Choice
		expected bool
	}{
		{
			name: "finish_reason_stop",
			choice: Choice{
				FinishReason: FinishReasonStop,
			},
			expected: true,
		},
		{
			name: "finish_reason_length",
			choice: Choice{
				FinishReason: FinishReasonLength,
			},
			expected: true,
		},
		{
			name: "finish_reason_tool_calls",
			choice: Choice{
				FinishReason: FinishReasonToolCalls,
			},
			expected: false,
		},
		{
			name: "no_finish_reason",
			choice: Choice{
				FinishReason: "",
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := tt.choice.IsComplete()
			if result != tt.expected {
				t.Errorf("IsComplete() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestChatResponse_RequiresToolExecution(t *testing.T) {
	tests := []struct {
		name     string
		response ChatResponse
		expected bool
	}{
		{
			name: "response_with_tool_calls",
			response: ChatResponse{
				Choices: []Choice{
					{
						FinishReason: FinishReasonToolCalls,
						Message: Message{
							Role: RoleAssistant,
							ToolCalls: []ToolCall{
								{
									ID:   "call_1",
									Type: "function",
									Function: ToolCallFunction{
										Name:      "search",
										Arguments: `{"query": "test"}`,
									},
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "response_without_tool_calls",
			response: ChatResponse{
				Choices: []Choice{
					{
						FinishReason: FinishReasonStop,
						Message: Message{
							Role:    RoleAssistant,
							Content: []MessageContent{NewTextContent("Hello")},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "multiple_choices_one_with_tools",
			response: ChatResponse{
				Choices: []Choice{
					{
						FinishReason: FinishReasonStop,
						Message: Message{
							Role:    RoleAssistant,
							Content: []MessageContent{NewTextContent("Hello")},
						},
					},
					{
						FinishReason: FinishReasonToolCalls,
						Message: Message{
							Role: RoleAssistant,
							ToolCalls: []ToolCall{
								{
									ID:   "call_1",
									Type: "function",
									Function: ToolCallFunction{
										Name:      "search",
										Arguments: `{"query": "test"}`,
									},
								},
							},
						},
					},
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := tt.response.RequiresToolExecution()
			if result != tt.expected {
				t.Errorf("RequiresToolExecution() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestChatResponse_GetToolCalls(t *testing.T) {
	tests := []struct {
		name            string
		response        ChatResponse
		expectedCount   int
		expectedToolIDs []string
	}{
		{
			name: "single_choice_with_tool_calls",
			response: ChatResponse{
				Choices: []Choice{
					{
						Message: Message{
							Role: RoleAssistant,
							ToolCalls: []ToolCall{
								{
									ID:   "call_1",
									Type: "function",
									Function: ToolCallFunction{
										Name:      "search",
										Arguments: `{"query": "test1"}`,
									},
								},
								{
									ID:   "call_2",
									Type: "function",
									Function: ToolCallFunction{
										Name:      "calculate",
										Arguments: `{"expression": "2+2"}`,
									},
								},
							},
						},
					},
				},
			},
			expectedCount:   2,
			expectedToolIDs: []string{"call_1", "call_2"},
		},
		{
			name: "multiple_choices_with_tool_calls",
			response: ChatResponse{
				Choices: []Choice{
					{
						Message: Message{
							Role: RoleAssistant,
							ToolCalls: []ToolCall{
								{
									ID:   "call_1",
									Type: "function",
									Function: ToolCallFunction{
										Name:      "search",
										Arguments: `{"query": "test1"}`,
									},
								},
							},
						},
					},
					{
						Message: Message{
							Role: RoleAssistant,
							ToolCalls: []ToolCall{
								{
									ID:   "call_2",
									Type: "function",
									Function: ToolCallFunction{
										Name:      "calculate",
										Arguments: `{"expression": "2+2"}`,
									},
								},
							},
						},
					},
				},
			},
			expectedCount:   2,
			expectedToolIDs: []string{"call_1", "call_2"},
		},
		{
			name: "no_tool_calls",
			response: ChatResponse{
				Choices: []Choice{
					{
						Message: Message{
							Role:    RoleAssistant,
							Content: []MessageContent{NewTextContent("Hello")},
						},
					},
				},
			},
			expectedCount:   0,
			expectedToolIDs: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			result := tt.response.GetToolCalls()

			if len(result) != tt.expectedCount {
				t.Errorf("GetToolCalls() returned %d tool calls, expected %d", len(result), tt.expectedCount)
			}

			for i, toolCall := range result {
				if i < len(tt.expectedToolIDs) && toolCall.ID != tt.expectedToolIDs[i] {
					t.Errorf("GetToolCalls()[%d].ID = %s, expected %s", i, toolCall.ID, tt.expectedToolIDs[i])
				}
			}
		})
	}
}
