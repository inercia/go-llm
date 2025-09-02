package llm

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"time"
)

// EnhancedTextContent provides enhanced JSON serialization for TextContent
// This extends the existing implementation with additional metadata
type EnhancedTextContentJSON struct {
	Type     MessageType `json:"type"`
	Text     string      `json:"text"`
	Encoding string      `json:"encoding,omitempty"`
	Version  string      `json:"version,omitempty"`
}

// EnhancedImageContentJSON provides enhanced JSON serialization for ImageContent with base64 support
type EnhancedImageContentJSON struct {
	Type     MessageType `json:"type"`
	Data     string      `json:"data,omitempty"`     // Base64 encoded binary data
	URL      string      `json:"url,omitempty"`      // URL reference for image
	MimeType string      `json:"mime_type"`          // Content type (required)
	Width    int         `json:"width,omitempty"`    // Image width in pixels
	Height   int         `json:"height,omitempty"`   // Image height in pixels
	Filename string      `json:"filename,omitempty"` // Original filename if available
	Encoding string      `json:"encoding,omitempty"` // Encoding type (base64)
	Version  string      `json:"version,omitempty"`  // Version for compatibility
}

// EnhancedFileContentJSON provides enhanced JSON serialization for FileContent with base64 support
type EnhancedFileContentJSON struct {
	Type     MessageType `json:"type"`
	Data     string      `json:"data,omitempty"`     // Base64 encoded binary data
	URL      string      `json:"url,omitempty"`      // URL reference for file
	MimeType string      `json:"mime_type"`          // Content type (required)
	Filename string      `json:"filename"`           // Original filename (required)
	Size     int64       `json:"size"`               // File size
	Encoding string      `json:"encoding,omitempty"` // Encoding type (base64)
	Version  string      `json:"version,omitempty"`  // Version for compatibility
}

// SerializationFormat represents different serialization formats
type SerializationFormat string

const (
	SerializationFormatStandard SerializationFormat = "standard" // Current format without binary data
	SerializationFormatEnhanced SerializationFormat = "enhanced" // Enhanced format with base64 binary data
	SerializationFormatCompact  SerializationFormat = "compact"  // Compact format optimized for size
)

const (
	// Current serialization version for compatibility
	CurrentSerializationVersion = "1.0"

	// Base64 encoding identifier
	Base64Encoding = "base64"
)

// SerializeMessage serializes a message to JSON with the specified format
func SerializeMessage(message Message, format SerializationFormat) ([]byte, error) {
	switch format {
	case SerializationFormatStandard:
		return message.MarshalJSON()
	case SerializationFormatEnhanced:
		return serializeMessageEnhanced(message)
	case SerializationFormatCompact:
		return serializeMessageCompact(message)
	default:
		return nil, fmt.Errorf("unsupported serialization format: %s", format)
	}
}

// DeserializeMessage deserializes JSON data to a Message, auto-detecting the format
func DeserializeMessage(data []byte) (Message, error) {
	var message Message
	if err := deserializeMessageEnhanced(data, &message); err == nil {
		return message, nil
	}

	// Fallback to standard format
	if err := message.UnmarshalJSON(data); err != nil {
		return Message{}, fmt.Errorf("failed to deserialize message in any supported format: %w", err)
	}

	return message, nil
}

// EstimateSerializedSize estimates the size of a message when serialized to JSON
func EstimateSerializedSize(message Message, format SerializationFormat) int64 {
	baseSize := int64(50) // More conservative base JSON overhead

	for _, content := range message.Content {
		if content == nil {
			continue
		}

		contentSize := content.Size()

		switch content.Type() {
		case MessageTypeText:
			// Text content + JSON overhead (more conservative)
			baseSize += contentSize + 30
		case MessageTypeImage, MessageTypeFile:
			switch format {
			case SerializationFormatStandard:
				// Only metadata, no binary data (more accurate)
				baseSize += 60
			case SerializationFormatEnhanced:
				// Base64 encoding increases size by ~33% + JSON overhead
				if contentSize > 0 {
					base64Size := int64(math.Ceil(float64(contentSize) * 4.0 / 3.0))
					baseSize += base64Size + 100 // + JSON overhead
				} else {
					baseSize += 100 // Just metadata for empty content
				}
			case SerializationFormatCompact:
				// Compressed base64 + minimal overhead
				if contentSize > 0 {
					base64Size := int64(math.Ceil(float64(contentSize) * 4.0 / 3.0))
					baseSize += base64Size + 30 // + minimal overhead
				} else {
					baseSize += 30
				}
			}
		}
	}

	return baseSize
}

// serializeMessageEnhanced serializes a message with enhanced format including base64 binary data
func serializeMessageEnhanced(message Message) ([]byte, error) {
	// Create enhanced message structure
	enhanced := struct {
		Role       MessageRole       `json:"role"`
		Content    []json.RawMessage `json:"content"`
		ToolCalls  []ToolCall        `json:"tool_calls,omitempty"`
		ToolCallID string            `json:"tool_call_id,omitempty"`
		Metadata   map[string]any    `json:"metadata,omitempty"`
		Version    string            `json:"version"`
	}{
		Role:       message.Role,
		ToolCalls:  message.ToolCalls,
		ToolCallID: message.ToolCallID,
		Metadata:   message.Metadata,
		Version:    CurrentSerializationVersion,
	}

	// Serialize each content item with enhanced format
	if len(message.Content) > 0 {
		enhanced.Content = make([]json.RawMessage, len(message.Content))
		for i, content := range message.Content {
			contentBytes, err := serializeContentEnhanced(content)
			if err != nil {
				return nil, fmt.Errorf("failed to serialize content item %d: %w", i, err)
			}
			enhanced.Content[i] = contentBytes
		}
	}

	return json.Marshal(enhanced)
}

// serializeContentEnhanced serializes individual content with base64 encoding for binary data
func serializeContentEnhanced(content MessageContent) ([]byte, error) {
	switch c := content.(type) {
	case *TextContent:
		enhanced := EnhancedTextContentJSON{
			Type:     c.Type(),
			Text:     c.Text,
			Encoding: "utf-8",
			Version:  CurrentSerializationVersion,
		}
		return json.Marshal(enhanced)

	case *ImageContent:
		enhanced := EnhancedImageContentJSON{
			Type:     c.Type(),
			URL:      c.URL,
			MimeType: c.MimeType,
			Width:    c.Width,
			Height:   c.Height,
			Filename: c.Filename,
			Version:  CurrentSerializationVersion,
		}

		// Add base64 encoded binary data if present and non-empty
		if c.HasData() && len(c.Data) > 0 {
			encoded := base64.StdEncoding.EncodeToString(c.Data)
			enhanced.Data = encoded
			enhanced.Encoding = Base64Encoding
		}

		return json.Marshal(enhanced)

	case *FileContent:
		enhanced := EnhancedFileContentJSON{
			Type:     c.Type(),
			URL:      c.URL,
			MimeType: c.MimeType,
			Filename: c.Filename,
			Size:     c.FileSize,
			Version:  CurrentSerializationVersion,
		}

		// Add base64 encoded binary data if present and non-empty
		if c.HasData() && len(c.Data) > 0 {
			encoded := base64.StdEncoding.EncodeToString(c.Data)
			enhanced.Data = encoded
			enhanced.Encoding = Base64Encoding
		}

		return json.Marshal(enhanced)

	default:
		return nil, fmt.Errorf("unsupported content type for enhanced serialization: %T", content)
	}
}

// deserializeMessageEnhanced deserializes JSON with enhanced format detection
func deserializeMessageEnhanced(data []byte, message *Message) error {
	// First try to detect if it's enhanced format by looking for version field
	var versionChecker struct {
		Version string `json:"version"`
	}

	if err := json.Unmarshal(data, &versionChecker); err != nil {
		return err
	}

	// If no version field, it's likely standard format
	if versionChecker.Version == "" {
		return fmt.Errorf("not enhanced format")
	}

	// Parse as enhanced format
	var enhanced struct {
		Role       MessageRole       `json:"role"`
		Content    []json.RawMessage `json:"content"`
		ToolCalls  []ToolCall        `json:"tool_calls,omitempty"`
		ToolCallID string            `json:"tool_call_id,omitempty"`
		Metadata   map[string]any    `json:"metadata,omitempty"`
		Version    string            `json:"version"`
	}

	if err := json.Unmarshal(data, &enhanced); err != nil {
		return err
	}

	// Set basic fields
	message.Role = enhanced.Role
	message.ToolCalls = enhanced.ToolCalls
	message.ToolCallID = enhanced.ToolCallID
	message.Metadata = enhanced.Metadata

	// Process enhanced content items
	if len(enhanced.Content) > 0 {
		message.Content = make([]MessageContent, 0, len(enhanced.Content))

		for i, contentBytes := range enhanced.Content {
			content, err := deserializeContentEnhanced(contentBytes)
			if err != nil {
				return fmt.Errorf("failed to deserialize enhanced content item %d: %w", i, err)
			}
			message.Content = append(message.Content, content)
		}
	}

	return nil
}

// deserializeContentEnhanced deserializes individual content from enhanced format
func deserializeContentEnhanced(data []byte) (MessageContent, error) {
	// First determine the content type
	var typeChecker struct {
		Type MessageType `json:"type"`
	}

	if err := json.Unmarshal(data, &typeChecker); err != nil {
		return nil, fmt.Errorf("failed to determine content type: %w", err)
	}

	switch typeChecker.Type {
	case MessageTypeText:
		var enhanced EnhancedTextContentJSON
		if err := json.Unmarshal(data, &enhanced); err != nil {
			return nil, err
		}
		return &TextContent{Text: enhanced.Text}, nil

	case MessageTypeImage:
		var enhanced EnhancedImageContentJSON
		if err := json.Unmarshal(data, &enhanced); err != nil {
			return nil, err
		}

		imageContent := &ImageContent{
			URL:      enhanced.URL,
			MimeType: enhanced.MimeType,
			Width:    enhanced.Width,
			Height:   enhanced.Height,
			Filename: enhanced.Filename,
		}

		// Decode base64 binary data if present
		if enhanced.Data != "" && enhanced.Encoding == Base64Encoding {
			decoded, err := base64.StdEncoding.DecodeString(enhanced.Data)
			if err != nil {
				return nil, fmt.Errorf("failed to decode base64 image data: %w", err)
			}
			imageContent.Data = decoded
		}

		return imageContent, nil

	case MessageTypeFile:
		var enhanced EnhancedFileContentJSON
		if err := json.Unmarshal(data, &enhanced); err != nil {
			return nil, err
		}

		fileContent := &FileContent{
			URL:      enhanced.URL,
			MimeType: enhanced.MimeType,
			Filename: enhanced.Filename,
			FileSize: enhanced.Size,
		}

		// Decode base64 binary data if present
		if enhanced.Data != "" && enhanced.Encoding == Base64Encoding {
			decoded, err := base64.StdEncoding.DecodeString(enhanced.Data)
			if err != nil {
				return nil, fmt.Errorf("failed to decode base64 file data: %w", err)
			}
			fileContent.Data = decoded

			// Verify size consistency
			if fileContent.FileSize != int64(len(decoded)) {
				return nil, fmt.Errorf("file size mismatch: expected %d, got %d", fileContent.FileSize, len(decoded))
			}
		}

		return fileContent, nil

	default:
		return nil, fmt.Errorf("unsupported content type: %s", typeChecker.Type)
	}
}

// serializeMessageCompact serializes a message in compact format for network efficiency
func serializeMessageCompact(message Message) ([]byte, error) {
	// Create compact message structure with minimal overhead
	compact := struct {
		R string            `json:"r"`           // Role (shortened)
		C []json.RawMessage `json:"c"`           // Content (shortened)
		T []ToolCall        `json:"t,omitempty"` // Tool calls (shortened)
		I string            `json:"i,omitempty"` // Tool call ID (shortened)
		M map[string]any    `json:"m,omitempty"` // Metadata (shortened)
		V string            `json:"v"`           // Version
	}{
		R: string(message.Role),
		T: message.ToolCalls,
		I: message.ToolCallID,
		M: message.Metadata,
		V: CurrentSerializationVersion,
	}

	// Serialize content in compact format
	if len(message.Content) > 0 {
		compact.C = make([]json.RawMessage, len(message.Content))
		for i, content := range message.Content {
			contentBytes, err := serializeContentCompact(content)
			if err != nil {
				return nil, fmt.Errorf("failed to serialize compact content item %d: %w", i, err)
			}
			compact.C[i] = contentBytes
		}
	}

	return json.Marshal(compact)
}

// serializeContentCompact serializes content in compact format
func serializeContentCompact(content MessageContent) ([]byte, error) {
	switch c := content.(type) {
	case *TextContent:
		compact := struct {
			T string `json:"t"` // Type
			X string `json:"x"` // Text (shortened)
		}{
			T: string(c.Type()),
			X: c.Text,
		}
		return json.Marshal(compact)

	case *ImageContent:
		compact := struct {
			T string `json:"t"`           // Type
			D string `json:"d,omitempty"` // Data (base64)
			U string `json:"u,omitempty"` // URL
			M string `json:"m"`           // MimeType
			W int    `json:"w,omitempty"` // Width
			H int    `json:"h,omitempty"` // Height
			F string `json:"f,omitempty"` // Filename
		}{
			T: string(c.Type()),
			U: c.URL,
			M: c.MimeType,
			W: c.Width,
			H: c.Height,
			F: c.Filename,
		}

		if c.HasData() {
			compact.D = base64.StdEncoding.EncodeToString(c.Data)
		}

		return json.Marshal(compact)

	case *FileContent:
		compact := struct {
			T string `json:"t"`           // Type
			D string `json:"d,omitempty"` // Data (base64)
			U string `json:"u,omitempty"` // URL
			M string `json:"m"`           // MimeType
			F string `json:"f"`           // Filename
			S int64  `json:"s"`           // Size
		}{
			T: string(c.Type()),
			U: c.URL,
			M: c.MimeType,
			F: c.Filename,
			S: c.FileSize,
		}

		if c.HasData() {
			compact.D = base64.StdEncoding.EncodeToString(c.Data)
		}

		return json.Marshal(compact)

	default:
		return nil, fmt.Errorf("unsupported content type for compact serialization: %T", content)
	}
}

// SerializationOptions holds options for controlling serialization behavior
type SerializationOptions struct {
	Format              SerializationFormat `json:"format"`
	IncludeBinaryData   bool                `json:"include_binary_data"`
	CompressBase64      bool                `json:"compress_base64"`
	MaxBinarySize       int64               `json:"max_binary_size"`
	UseURLForLargeFiles bool                `json:"use_url_for_large_files"`
}

// SerializeMessageWithOptions serializes a message with custom options
func SerializeMessageWithOptions(message Message, options SerializationOptions) ([]byte, error) {
	switch options.Format {
	case SerializationFormatEnhanced:
		return serializeMessageWithEnhancedOptions(message, options)
	case SerializationFormatCompact:
		return serializeMessageCompact(message)
	default:
		return message.MarshalJSON()
	}
}

// serializeMessageWithEnhancedOptions serializes with enhanced options
func serializeMessageWithEnhancedOptions(message Message, options SerializationOptions) ([]byte, error) {
	enhanced := struct {
		Role       MessageRole          `json:"role"`
		Content    []json.RawMessage    `json:"content"`
		ToolCalls  []ToolCall           `json:"tool_calls,omitempty"`
		ToolCallID string               `json:"tool_call_id,omitempty"`
		Metadata   map[string]any       `json:"metadata,omitempty"`
		Version    string               `json:"version"`
		Options    SerializationOptions `json:"options,omitempty"`
	}{
		Role:       message.Role,
		ToolCalls:  message.ToolCalls,
		ToolCallID: message.ToolCallID,
		Metadata:   message.Metadata,
		Version:    CurrentSerializationVersion,
		Options:    options,
	}

	// Process content with options
	if len(message.Content) > 0 {
		enhanced.Content = make([]json.RawMessage, len(message.Content))
		for i, content := range message.Content {
			contentBytes, err := serializeContentWithOptions(content, options)
			if err != nil {
				return nil, fmt.Errorf("failed to serialize content item %d with options: %w", i, err)
			}
			enhanced.Content[i] = contentBytes
		}
	}

	return json.Marshal(enhanced)
}

// serializeContentWithOptions serializes content respecting the provided options
func serializeContentWithOptions(content MessageContent, options SerializationOptions) ([]byte, error) {
	switch c := content.(type) {
	case *TextContent:
		return serializeContentEnhanced(content)

	case *ImageContent:
		enhanced := EnhancedImageContentJSON{
			Type:     c.Type(),
			URL:      c.URL,
			MimeType: c.MimeType,
			Width:    c.Width,
			Height:   c.Height,
			Filename: c.Filename,
			Version:  CurrentSerializationVersion,
		}

		// Handle binary data based on options
		if c.HasData() && len(c.Data) > 0 && options.IncludeBinaryData {
			if c.Size() <= options.MaxBinarySize {
				enhanced.Data = base64.StdEncoding.EncodeToString(c.Data)
				enhanced.Encoding = Base64Encoding
			} else if options.UseURLForLargeFiles {
				// In production, this would store to a URL-accessible location
				enhanced.URL = fmt.Sprintf("temp://large-content-%d", time.Now().UnixNano())
			}
		}

		return json.Marshal(enhanced)

	case *FileContent:
		enhanced := EnhancedFileContentJSON{
			Type:     c.Type(),
			URL:      c.URL,
			MimeType: c.MimeType,
			Filename: c.Filename,
			Size:     c.FileSize,
			Version:  CurrentSerializationVersion,
		}

		// Handle binary data based on options
		if c.HasData() && len(c.Data) > 0 && options.IncludeBinaryData {
			if c.Size() <= options.MaxBinarySize {
				enhanced.Data = base64.StdEncoding.EncodeToString(c.Data)
				enhanced.Encoding = Base64Encoding
			} else if options.UseURLForLargeFiles {
				// In production, this would store to a URL-accessible location
				enhanced.URL = fmt.Sprintf("temp://large-content-%d", time.Now().UnixNano())
			}
		}

		return json.Marshal(enhanced)

	default:
		return nil, fmt.Errorf("unsupported content type: %T", content)
	}
}

// ValidateSerializedData validates that serialized data is well-formed JSON
func ValidateSerializedData(data []byte) error {
	var temp interface{}
	return json.Unmarshal(data, &temp)
}

// GetSerializedVersion attempts to extract the version from serialized data
func GetSerializedVersion(data []byte) (string, error) {
	var versionChecker struct {
		Version string `json:"version"`
	}

	if err := json.Unmarshal(data, &versionChecker); err != nil {
		return "", err
	}

	if versionChecker.Version == "" {
		return "legacy", nil // No version means legacy format
	}

	return versionChecker.Version, nil
}

// IsEnhancedFormat checks if the serialized data uses the enhanced format
func IsEnhancedFormat(data []byte) bool {
	version, err := GetSerializedVersion(data)
	return err == nil && version != "legacy"
}

// ConvertSerializationFormat converts between different serialization formats
func ConvertSerializationFormat(data []byte, targetFormat SerializationFormat) ([]byte, error) {
	// First deserialize the message
	message, err := DeserializeMessage(data)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize for conversion: %w", err)
	}

	// Re-serialize in target format
	return SerializeMessage(message, targetFormat)
}

// SerializationStats provides statistics about serialized content
type SerializationStats struct {
	OriginalSize     int64               `json:"original_size"`
	SerializedSize   int64               `json:"serialized_size"`
	CompressionRatio float64             `json:"compression_ratio"`
	Base64Overhead   int64               `json:"base64_overhead"`
	ContentTypes     map[MessageType]int `json:"content_types"`
}

// AnalyzeSerialization analyzes the serialization efficiency of a message
func AnalyzeSerialization(message Message, format SerializationFormat) (SerializationStats, error) {
	originalSize := message.TotalSize()

	serialized, err := SerializeMessage(message, format)
	if err != nil {
		return SerializationStats{}, err
	}

	serializedSize := int64(len(serialized))

	// Count content types
	contentTypes := make(map[MessageType]int)
	for _, content := range message.Content {
		if content != nil {
			contentTypes[content.Type()]++
		}
	}

	// Calculate base64 overhead for binary content
	base64Overhead := int64(0)
	for _, content := range message.Content {
		if content != nil && (content.Type() == MessageTypeImage || content.Type() == MessageTypeFile) {
			contentSize := content.Size()
			base64Size := int64(math.Ceil(float64(contentSize) * 4.0 / 3.0))
			base64Overhead += base64Size - contentSize
		}
	}

	var compressionRatio float64
	if originalSize > 0 {
		compressionRatio = float64(serializedSize) / float64(originalSize)
	}

	return SerializationStats{
		OriginalSize:     originalSize,
		SerializedSize:   serializedSize,
		CompressionRatio: compressionRatio,
		Base64Overhead:   base64Overhead,
		ContentTypes:     contentTypes,
	}, nil
}
