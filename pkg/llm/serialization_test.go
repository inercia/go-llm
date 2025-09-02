package llm

import (
	"encoding/base64"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestSerializeMessage_Standard(t *testing.T) {
	// Test standard format (existing implementation)
	message := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("Hello, World!"),
		},
	}

	data, err := SerializeMessage(message, SerializationFormatStandard)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	if len(data) == 0 {
		t.Error("Expected serialized data, got empty")
	}

	// Verify it's valid JSON
	if err := ValidateSerializedData(data); err != nil {
		t.Errorf("Invalid JSON produced: %v", err)
	}
}

func TestSerializeMessage_Enhanced(t *testing.T) {
	// Create message with binary content
	imageData := []byte("fake image binary data")
	fileData := []byte("fake file binary data")

	message := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("Hello, World!"),
			NewImageContentFromBytes(imageData, "image/png"),
			NewFileContentFromBytes(fileData, "test.txt", "text/plain"),
		},
	}

	data, err := SerializeMessage(message, SerializationFormatEnhanced)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	if len(data) == 0 {
		t.Error("Expected serialized data, got empty")
	}

	// Verify it's valid JSON
	if err := ValidateSerializedData(data); err != nil {
		t.Errorf("Invalid JSON produced: %v", err)
	}

	// Verify version is included
	version, err := GetSerializedVersion(data)
	if err != nil {
		t.Errorf("Failed to get version: %v", err)
	}
	if version != CurrentSerializationVersion {
		t.Errorf("Expected version %s, got %s", CurrentSerializationVersion, version)
	}

	// Verify binary data is base64 encoded
	dataStr := string(data)
	expectedImageB64 := base64.StdEncoding.EncodeToString(imageData)
	expectedFileB64 := base64.StdEncoding.EncodeToString(fileData)

	if !strings.Contains(dataStr, expectedImageB64) {
		t.Error("Expected base64 encoded image data in JSON")
	}
	if !strings.Contains(dataStr, expectedFileB64) {
		t.Error("Expected base64 encoded file data in JSON")
	}
}

func TestSerializeMessage_Compact(t *testing.T) {
	message := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("Hello"),
			NewImageContentFromBytes([]byte("img"), "image/png"),
		},
	}

	data, err := SerializeMessage(message, SerializationFormatCompact)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	// Verify compact format uses shorter field names
	dataStr := string(data)
	if !strings.Contains(dataStr, `"r":`) { // Role shortened to 'r'
		t.Error("Expected compact format with shortened field names")
	}
	if !strings.Contains(dataStr, `"c":`) { // Content shortened to 'c'
		t.Error("Expected compact format with shortened content field")
	}
}

func TestDeserializeMessage_RoundTrip(t *testing.T) {
	// Test round-trip serialization fidelity
	imageData := make([]byte, 100)
	for i := range imageData {
		imageData[i] = byte(i % 256)
	}

	original := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("Test message"),
			NewImageContentFromBytes(imageData, "image/png"),
			NewFileContentFromBytes([]byte("file content"), "test.txt", "text/plain"),
		},
		Metadata: map[string]any{
			"test": "metadata",
		},
	}

	// Test enhanced format round-trip
	serialized, err := SerializeMessage(original, SerializationFormatEnhanced)
	if err != nil {
		t.Fatalf("Failed to serialize: %v", err)
	}

	deserialized, err := DeserializeMessage(serialized)
	if err != nil {
		t.Fatalf("Failed to deserialize: %v", err)
	}

	// Verify basic fields
	if deserialized.Role != original.Role {
		t.Errorf("Role mismatch: expected %s, got %s", original.Role, deserialized.Role)
	}

	if len(deserialized.Content) != len(original.Content) {
		t.Errorf("Content count mismatch: expected %d, got %d", len(original.Content), len(deserialized.Content))
	}

	// Verify text content
	if textContent, ok := deserialized.Content[0].(*TextContent); ok {
		originalText := original.Content[0].(*TextContent)
		if textContent.Text != originalText.Text {
			t.Errorf("Text content mismatch: expected %s, got %s", originalText.Text, textContent.Text)
		}
	} else {
		t.Error("Expected first content to be TextContent")
	}

	// Verify image content and binary data
	if imageContent, ok := deserialized.Content[1].(*ImageContent); ok {
		originalImage := original.Content[1].(*ImageContent)
		if imageContent.MimeType != originalImage.MimeType {
			t.Errorf("Image MIME type mismatch: expected %s, got %s", originalImage.MimeType, imageContent.MimeType)
		}
		if !reflect.DeepEqual(imageContent.Data, originalImage.Data) {
			t.Error("Image binary data mismatch")
		}
	} else {
		t.Error("Expected second content to be ImageContent")
	}
}

func TestEstimateSerializedSize(t *testing.T) {
	tests := []struct {
		name    string
		message Message
		format  SerializationFormat
		minSize int64
	}{
		{
			name: "text only message",
			message: Message{
				Content: []MessageContent{
					NewTextContent("Hello, World!"),
				},
			},
			format:  SerializationFormatStandard,
			minSize: 50, // More realistic base + text + overhead
		},
		{
			name: "binary content enhanced",
			message: Message{
				Content: []MessageContent{
					NewImageContentFromBytes(make([]byte, 1000), "image/png"),
				},
			},
			format:  SerializationFormatEnhanced,
			minSize: 1400, // Base64 overhead included
		},
		{
			name: "binary content standard",
			message: Message{
				Content: []MessageContent{
					NewImageContentFromBytes(make([]byte, 1000), "image/png"),
				},
			},
			format:  SerializationFormatStandard,
			minSize: 50, // Metadata only, no binary
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			estimated := EstimateSerializedSize(tt.message, tt.format)
			if estimated < tt.minSize {
				t.Errorf("Estimated size %d is less than expected minimum %d", estimated, tt.minSize)
			}

			// Verify estimate is reasonably accurate
			actual, err := SerializeMessage(tt.message, tt.format)
			if err != nil {
				t.Fatalf("Failed to serialize for verification: %v", err)
			}

			actualSize := int64(len(actual))
			difference := estimated - actualSize
			if difference < 0 {
				difference = -difference
			}

			// Allow up to 100% variance in estimation (it's an estimate and can vary significantly)
			maxVariance := actualSize
			if maxVariance < 100 {
				maxVariance = 100 // Minimum tolerance for small messages
			}
			if difference > maxVariance {
				t.Errorf("Size estimate %d differs too much from actual %d (difference: %d)",
					estimated, actualSize, difference)
			}
		})
	}
}

func TestSerializeContentEnhanced_TextContent(t *testing.T) {
	content := NewTextContent("Hello, World!")

	data, err := serializeContentEnhanced(content)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	// Verify it contains type and text fields
	dataStr := string(data)
	if !strings.Contains(dataStr, `"type":"text"`) {
		t.Error("Expected type field in serialized text content")
	}
	if !strings.Contains(dataStr, `"text":"Hello, World!"`) {
		t.Error("Expected text field in serialized text content")
	}
	if !strings.Contains(dataStr, `"version":"1.0"`) {
		t.Error("Expected version field in serialized text content")
	}
}

func TestSerializeContentEnhanced_ImageContent(t *testing.T) {
	imageData := []byte("fake image data")
	content := NewImageContentFromBytes(imageData, "image/png")
	content.Width = 100
	content.Height = 200
	content.Filename = "test.png"

	data, err := serializeContentEnhanced(content)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	dataStr := string(data)

	// Verify all fields are present
	if !strings.Contains(dataStr, `"type":"image"`) {
		t.Error("Expected type field")
	}
	if !strings.Contains(dataStr, `"mime_type":"image/png"`) {
		t.Error("Expected mime_type field")
	}
	if !strings.Contains(dataStr, `"width":100`) {
		t.Error("Expected width field")
	}
	if !strings.Contains(dataStr, `"height":200`) {
		t.Error("Expected height field")
	}
	if !strings.Contains(dataStr, `"filename":"test.png"`) {
		t.Error("Expected filename field")
	}
	if !strings.Contains(dataStr, `"encoding":"base64"`) {
		t.Error("Expected encoding field")
	}

	// Verify base64 encoded data is present
	expectedB64 := base64.StdEncoding.EncodeToString(imageData)
	if !strings.Contains(dataStr, expectedB64) {
		t.Error("Expected base64 encoded image data")
	}
}

func TestSerializeContentEnhanced_FileContent(t *testing.T) {
	fileData := []byte("fake file content")
	content := NewFileContentFromBytes(fileData, "test.txt", "text/plain")

	data, err := serializeContentEnhanced(content)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	dataStr := string(data)

	// Verify all fields are present
	if !strings.Contains(dataStr, `"type":"file"`) {
		t.Error("Expected type field")
	}
	if !strings.Contains(dataStr, `"mime_type":"text/plain"`) {
		t.Error("Expected mime_type field")
	}
	if !strings.Contains(dataStr, `"filename":"test.txt"`) {
		t.Error("Expected filename field")
	}
	if !strings.Contains(dataStr, `"encoding":"base64"`) {
		t.Error("Expected encoding field")
	}

	// Verify base64 encoded data is present
	expectedB64 := base64.StdEncoding.EncodeToString(fileData)
	if !strings.Contains(dataStr, expectedB64) {
		t.Error("Expected base64 encoded file data")
	}
}

func TestDeserializeContentEnhanced_ImageContent(t *testing.T) {
	imageData := []byte("test image data")
	base64Data := base64.StdEncoding.EncodeToString(imageData)

	jsonData := fmt.Sprintf(`{
		"type": "image",
		"data": "%s",
		"mime_type": "image/png", 
		"width": 100,
		"height": 200,
		"filename": "test.png",
		"encoding": "base64",
		"version": "1.0"
	}`, base64Data)

	content, err := deserializeContentEnhanced([]byte(jsonData))
	if err != nil {
		t.Fatalf("Failed to deserialize: %v", err)
	}

	imageContent, ok := content.(*ImageContent)
	if !ok {
		t.Fatal("Expected ImageContent")
	}

	if imageContent.MimeType != "image/png" {
		t.Errorf("Expected MIME type image/png, got %s", imageContent.MimeType)
	}
	if imageContent.Width != 100 {
		t.Errorf("Expected width 100, got %d", imageContent.Width)
	}
	if imageContent.Height != 200 {
		t.Errorf("Expected height 200, got %d", imageContent.Height)
	}
	if imageContent.Filename != "test.png" {
		t.Errorf("Expected filename test.png, got %s", imageContent.Filename)
	}
	if !reflect.DeepEqual(imageContent.Data, imageData) {
		t.Error("Binary data mismatch after deserialization")
	}
}

func TestDeserializeContentEnhanced_FileContent(t *testing.T) {
	fileData := []byte("test file content")
	base64Data := base64.StdEncoding.EncodeToString(fileData)

	jsonData := fmt.Sprintf(`{
		"type": "file",
		"data": "%s",
		"mime_type": "text/plain",
		"filename": "test.txt",
		"size": %d,
		"encoding": "base64",
		"version": "1.0"
	}`, base64Data, len(fileData))

	content, err := deserializeContentEnhanced([]byte(jsonData))
	if err != nil {
		t.Fatalf("Failed to deserialize: %v", err)
	}

	fileContent, ok := content.(*FileContent)
	if !ok {
		t.Fatal("Expected FileContent")
	}

	if fileContent.MimeType != "text/plain" {
		t.Errorf("Expected MIME type text/plain, got %s", fileContent.MimeType)
	}
	if fileContent.Filename != "test.txt" {
		t.Errorf("Expected filename test.txt, got %s", fileContent.Filename)
	}
	if fileContent.FileSize != int64(len(fileData)) {
		t.Errorf("Expected size %d, got %d", len(fileData), fileContent.FileSize)
	}
	if !reflect.DeepEqual(fileContent.Data, fileData) {
		t.Error("Binary data mismatch after deserialization")
	}
}

func TestDeserializeMessage_LegacyFormat(t *testing.T) {
	// Create legacy format JSON (without version field)
	legacyJSON := `{
		"role": "user",
		"content": [
			{
				"type": "text",
				"text": "Hello, World!"
			}
		]
	}`

	message, err := DeserializeMessage([]byte(legacyJSON))
	if err != nil {
		t.Errorf("Failed to deserialize legacy format: %v", err)
	}

	if message.Role != RoleUser {
		t.Errorf("Expected role user, got %s", message.Role)
	}

	if len(message.Content) != 1 {
		t.Errorf("Expected 1 content item, got %d", len(message.Content))
	}
}

func TestSerializeMessageWithOptions(t *testing.T) {
	largeData := make([]byte, 1000)
	smallData := make([]byte, 100)

	message := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewImageContentFromBytes(largeData, "image/png"),
			NewFileContentFromBytes(smallData, "small.txt", "text/plain"),
		},
	}

	options := SerializationOptions{
		Format:              SerializationFormatEnhanced,
		IncludeBinaryData:   true,
		MaxBinarySize:       500, // Only small file should be included
		UseURLForLargeFiles: true,
	}

	data, err := SerializeMessageWithOptions(message, options)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	dataStr := string(data)

	// Large image should have temp URL, not base64 data
	if strings.Contains(dataStr, base64.StdEncoding.EncodeToString(largeData)) {
		t.Error("Large image data should not be included as base64")
	}
	if !strings.Contains(dataStr, "temp://large-content") {
		t.Error("Large image should have temp URL")
	}

	// Small file should have base64 data
	if !strings.Contains(dataStr, base64.StdEncoding.EncodeToString(smallData)) {
		t.Error("Small file data should be included as base64")
	}
}

func TestAnalyzeSerialization(t *testing.T) {
	imageData := make([]byte, 1000)
	message := Message{
		Content: []MessageContent{
			NewTextContent("Hello"),
			NewImageContentFromBytes(imageData, "image/png"),
		},
	}

	stats, err := AnalyzeSerialization(message, SerializationFormatEnhanced)
	if err != nil {
		t.Errorf("Failed to analyze serialization: %v", err)
	}

	if stats.OriginalSize == 0 {
		t.Error("Expected original size > 0")
	}
	if stats.SerializedSize == 0 {
		t.Error("Expected serialized size > 0")
	}
	if stats.CompressionRatio == 0 {
		t.Error("Expected compression ratio > 0")
	}
	if stats.Base64Overhead == 0 {
		t.Error("Expected base64 overhead > 0 for binary content")
	}

	// Verify content type counting
	if stats.ContentTypes[MessageTypeText] != 1 {
		t.Errorf("Expected 1 text content, got %d", stats.ContentTypes[MessageTypeText])
	}
	if stats.ContentTypes[MessageTypeImage] != 1 {
		t.Errorf("Expected 1 image content, got %d", stats.ContentTypes[MessageTypeImage])
	}
}

func TestConvertSerializationFormat(t *testing.T) {
	message := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("Test"),
			NewImageContentFromBytes([]byte("image data"), "image/png"),
		},
	}

	// Serialize in enhanced format
	enhanced, err := SerializeMessage(message, SerializationFormatEnhanced)
	if err != nil {
		t.Fatalf("Failed to serialize enhanced: %v", err)
	}

	// Convert to compact format
	compact, err := ConvertSerializationFormat(enhanced, SerializationFormatCompact)
	if err != nil {
		t.Errorf("Failed to convert format: %v", err)
	}

	// Verify compact format characteristics
	compactStr := string(compact)
	if !strings.Contains(compactStr, `"r":`) {
		t.Error("Expected compact format field names")
	}

	// Should be smaller than enhanced format
	if len(compact) >= len(enhanced) {
		t.Error("Compact format should be smaller than enhanced format")
	}
}

func TestSerializationFormats_ErrorHandling(t *testing.T) {
	message := Message{
		Content: []MessageContent{
			NewTextContent("test"),
		},
	}

	// Test unsupported format
	_, err := SerializeMessage(message, "unsupported")
	if err == nil {
		t.Error("Expected error for unsupported format")
	}

	// Test malformed JSON deserialization
	malformedJSON := `{"role": "user", "content": [{"type": "text", "text": "unclosed string}`
	_, err = DeserializeMessage([]byte(malformedJSON))
	if err == nil {
		t.Error("Expected error for malformed JSON")
	}
}

func TestBase64EncodingDecoding(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{
			name: "simple data",
			data: []byte("Hello, World!"),
		},
		{
			name: "binary data",
			data: []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}, // PNG header
		},
		{
			name: "empty data",
			data: []byte{},
		},
		{
			name: "large data",
			data: make([]byte, 10000),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test round-trip through image content
			imageContent := NewImageContentFromBytes(tt.data, "image/png")

			serialized, err := serializeContentEnhanced(imageContent)
			if err != nil {
				t.Fatalf("Failed to serialize: %v", err)
			}

			deserialized, err := deserializeContentEnhanced(serialized)
			if err != nil {
				t.Fatalf("Failed to deserialize: %v", err)
			}

			deserializedImage, ok := deserialized.(*ImageContent)
			if !ok {
				t.Fatal("Expected ImageContent")
			}

			// For empty data, we expect nil/empty since we don't serialize empty binary data
			if len(tt.data) == 0 {
				if len(deserializedImage.Data) != 0 {
					t.Error("Expected empty data to remain empty")
				}
			} else {
				if !reflect.DeepEqual(deserializedImage.Data, tt.data) {
					t.Error("Binary data corrupted during base64 round-trip")
				}
			}
		})
	}
}

func TestVersionCompatibility(t *testing.T) {
	// Test that we can handle different version formats
	tests := []struct {
		name    string
		jsonStr string
		isValid bool
	}{
		{
			name: "current version",
			jsonStr: `{
				"role": "user",
				"content": [],
				"version": "1.0"
			}`,
			isValid: true,
		},
		{
			name: "legacy format (no version)",
			jsonStr: `{
				"role": "user", 
				"content": []
			}`,
			isValid: true,
		},
		{
			name: "future version",
			jsonStr: `{
				"role": "user",
				"content": [],
				"version": "2.0"
			}`,
			isValid: true, // Should still work
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := DeserializeMessage([]byte(tt.jsonStr))
			if tt.isValid && err != nil {
				t.Errorf("Expected valid deserialization, got error: %v", err)
			}
			if !tt.isValid && err == nil {
				t.Error("Expected deserialization error, got success")
			}
		})
	}
}

func TestMixedContentTypes(t *testing.T) {
	// Test message with all content types
	message := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("Hello"),
			NewImageContentFromBytes([]byte("image"), "image/png"),
			NewFileContentFromBytes([]byte("file"), "test.txt", "text/plain"),
			NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
		},
	}

	// Test enhanced serialization
	serialized, err := SerializeMessage(message, SerializationFormatEnhanced)
	if err != nil {
		t.Fatalf("Failed to serialize mixed content: %v", err)
	}

	// Test deserialization
	deserialized, err := DeserializeMessage(serialized)
	if err != nil {
		t.Fatalf("Failed to deserialize mixed content: %v", err)
	}

	if len(deserialized.Content) != 4 {
		t.Errorf("Expected 4 content items, got %d", len(deserialized.Content))
	}

	// Verify each content type
	expectedTypes := []MessageType{MessageTypeText, MessageTypeImage, MessageTypeFile, MessageTypeImage}
	for i, content := range deserialized.Content {
		if content.Type() != expectedTypes[i] {
			t.Errorf("Content %d: expected type %s, got %s", i, expectedTypes[i], content.Type())
		}
	}
}

func TestSerializationEdgeCases(t *testing.T) {
	t.Run("empty message", func(t *testing.T) {
		message := Message{Role: RoleUser, Content: []MessageContent{}}

		data, err := SerializeMessage(message, SerializationFormatEnhanced)
		if err != nil {
			t.Errorf("Failed to serialize empty message: %v", err)
		}

		deserialized, err := DeserializeMessage(data)
		if err != nil {
			t.Errorf("Failed to deserialize empty message: %v", err)
		}

		if len(deserialized.Content) != 0 {
			t.Errorf("Expected empty content, got %d items", len(deserialized.Content))
		}
	})

	t.Run("special characters in text", func(t *testing.T) {
		specialText := "Special chars: 🌟 \n\t\r\"'\\"
		message := Message{
			Content: []MessageContent{
				NewTextContent(specialText),
			},
		}

		serialized, err := SerializeMessage(message, SerializationFormatEnhanced)
		if err != nil {
			t.Errorf("Failed to serialize special characters: %v", err)
		}

		deserialized, err := DeserializeMessage(serialized)
		if err != nil {
			t.Errorf("Failed to deserialize special characters: %v", err)
		}

		if textContent, ok := deserialized.Content[0].(*TextContent); ok {
			if textContent.Text != specialText {
				t.Errorf("Special characters corrupted: expected %q, got %q", specialText, textContent.Text)
			}
		}
	})

	t.Run("invalid base64 data", func(t *testing.T) {
		invalidJSON := `{
			"type": "image",
			"data": "invalid-base64-data!!!",
			"mime_type": "image/png",
			"encoding": "base64"
		}`

		_, err := deserializeContentEnhanced([]byte(invalidJSON))
		if err == nil {
			t.Error("Expected error for invalid base64 data")
		}
	})

	t.Run("size mismatch in file content", func(t *testing.T) {
		fileData := []byte("test")
		base64Data := base64.StdEncoding.EncodeToString(fileData)

		invalidJSON := fmt.Sprintf(`{
			"type": "file",
			"data": "%s",
			"mime_type": "text/plain",
			"filename": "test.txt",
			"size": 999,
			"encoding": "base64"
		}`, base64Data)

		_, err := deserializeContentEnhanced([]byte(invalidJSON))
		if err == nil {
			t.Error("Expected error for file size mismatch")
		}
	})
}

func TestGetSerializedVersion(t *testing.T) {
	tests := []struct {
		name            string
		jsonData        string
		expectedVersion string
		expectError     bool
	}{
		{
			name:            "version 1.0",
			jsonData:        `{"version": "1.0", "role": "user"}`,
			expectedVersion: "1.0",
			expectError:     false,
		},
		{
			name:            "no version (legacy)",
			jsonData:        `{"role": "user", "content": []}`,
			expectedVersion: "legacy",
			expectError:     false,
		},
		{
			name:            "invalid JSON",
			jsonData:        `{"version": "1.0", "invalid`,
			expectedVersion: "",
			expectError:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			version, err := GetSerializedVersion([]byte(tt.jsonData))

			if tt.expectError && err == nil {
				t.Error("Expected error, got nil")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error, got: %v", err)
			}
			if version != tt.expectedVersion {
				t.Errorf("Expected version %s, got %s", tt.expectedVersion, version)
			}
		})
	}
}

func TestIsEnhancedFormat(t *testing.T) {
	enhancedJSON := `{"version": "1.0", "role": "user"}`
	legacyJSON := `{"role": "user", "content": []}`

	if !IsEnhancedFormat([]byte(enhancedJSON)) {
		t.Error("Expected enhanced format to be detected")
	}

	if IsEnhancedFormat([]byte(legacyJSON)) {
		t.Error("Expected legacy format to be detected")
	}
}

// Benchmark tests
func BenchmarkSerializeMessage_Enhanced(b *testing.B) {
	imageData := make([]byte, 1024)
	message := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("Benchmark test"),
			NewImageContentFromBytes(imageData, "image/png"),
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := SerializeMessage(message, SerializationFormatEnhanced)
		if err != nil {
			b.Errorf("Serialization failed: %v", err)
		}
	}
}

func BenchmarkDeserializeMessage(b *testing.B) {
	message := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("Benchmark test"),
			NewImageContentFromBytes(make([]byte, 1024), "image/png"),
		},
	}

	serialized, err := SerializeMessage(message, SerializationFormatEnhanced)
	if err != nil {
		b.Fatalf("Failed to prepare test data: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := DeserializeMessage(serialized)
		if err != nil {
			b.Errorf("Deserialization failed: %v", err)
		}
	}
}

func BenchmarkEstimateSerializedSize(b *testing.B) {
	message := Message{
		Content: []MessageContent{
			NewTextContent("Test message"),
			NewImageContentFromBytes(make([]byte, 2048), "image/png"),
			NewFileContentFromBytes(make([]byte, 1024), "test.txt", "text/plain"),
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EstimateSerializedSize(message, SerializationFormatEnhanced)
	}
}
