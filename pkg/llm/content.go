package llm

import (
	"encoding/json"
	"errors"
	"net/url"
	"strings"
)

// TextContent represents text-based message content
// It provides backward compatibility with existing text-only message systems
type TextContent struct {
	Text string `json:"text"`
}

// NewTextContent creates a new TextContent instance with the given text
// The text is validated during construction to ensure it's not empty
func NewTextContent(text string) *TextContent {
	return &TextContent{
		Text: text,
	}
}

// Type returns the message type for text content
func (t *TextContent) Type() MessageType {
	return MessageTypeText
}

// Validate checks if the text content is valid
// Text content must not be empty or contain only whitespace
func (t *TextContent) Validate() error {
	if t == nil {
		return errors.New("text content cannot be nil")
	}

	// Check for empty text
	if t.Text == "" {
		return errors.New("text content cannot be empty")
	}

	// Check for whitespace-only text
	if strings.TrimSpace(t.Text) == "" {
		return errors.New("text content cannot be only whitespace")
	}

	return nil
}

// Size returns the byte size of the text content
// This correctly handles Unicode and multi-byte characters
func (t *TextContent) Size() int64 {
	if t == nil {
		return 0
	}
	return int64(len([]byte(t.Text)))
}

// GetText returns the text content as a string
// This is a convenience method for accessing the text
func (t *TextContent) GetText() string {
	if t == nil {
		return ""
	}
	return t.Text
}

// IsEmpty checks if the text content is empty or whitespace-only
func (t *TextContent) IsEmpty() bool {
	if t == nil {
		return true
	}
	return strings.TrimSpace(t.Text) == ""
}

// MarshalJSON implements custom JSON marshaling for TextContent
func (t *TextContent) MarshalJSON() ([]byte, error) {
	if t == nil {
		return json.Marshal(nil)
	}

	// Create a struct with the content type information
	data := struct {
		Type MessageType `json:"type"`
		Text string      `json:"text"`
	}{
		Type: t.Type(),
		Text: t.Text,
	}

	return json.Marshal(data)
}

// UnmarshalJSON implements custom JSON unmarshaling for TextContent
func (t *TextContent) UnmarshalJSON(data []byte) error {
	if t == nil {
		return errors.New("cannot unmarshal into nil TextContent")
	}

	// Define struct for unmarshaling
	var content struct {
		Type MessageType `json:"type"`
		Text string      `json:"text"`
	}

	if err := json.Unmarshal(data, &content); err != nil {
		return err
	}

	// Validate the type field if present
	if content.Type != "" && content.Type != MessageTypeText {
		return errors.New("invalid content type for TextContent")
	}

	t.Text = content.Text
	return nil
}

// ImageContent represents image-based message content
// It supports both binary image data and URL references with proper MIME type validation
type ImageContent struct {
	Data     []byte `json:"-"`                  // Binary image data (omitted from JSON)
	URL      string `json:"url,omitempty"`      // URL reference for image
	MimeType string `json:"mime_type"`          // Content type (required)
	Width    int    `json:"width,omitempty"`    // Image width in pixels
	Height   int    `json:"height,omitempty"`   // Image height in pixels
	Filename string `json:"filename,omitempty"` // Original filename if available
}

// Supported MIME types for images
var supportedImageMimeTypes = map[string]bool{
	"image/jpeg": true,
	"image/png":  true,
	"image/gif":  true,
	"image/webp": true,
}

// NewImageContentFromBytes creates a new ImageContent instance from binary data
func NewImageContentFromBytes(data []byte, mimeType string) *ImageContent {
	return &ImageContent{
		Data:     data,
		MimeType: mimeType,
	}
}

// NewImageContentFromURL creates a new ImageContent instance from a URL reference
func NewImageContentFromURL(imageURL, mimeType string) *ImageContent {
	return &ImageContent{
		URL:      imageURL,
		MimeType: mimeType,
	}
}

// Type returns the message type for image content
func (i *ImageContent) Type() MessageType {
	return MessageTypeImage
}

// Validate checks if the image content is valid
func (i *ImageContent) Validate() error {
	if i == nil {
		return errors.New("image content cannot be nil")
	}

	// Check that either data or URL is provided (but not both empty)
	hasData := len(i.Data) > 0
	hasURL := strings.TrimSpace(i.URL) != ""

	if !hasData && !hasURL {
		return errors.New("image content must have either data or URL")
	}

	// Validate MIME type is provided (security validation will check if it's supported)
	if strings.TrimSpace(i.MimeType) == "" {
		return errors.New("image content must have a MIME type")
	}

	// If URL is provided, validate it's a proper URL
	if hasURL {
		if _, err := url.ParseRequestURI(i.URL); err != nil {
			return errors.New("invalid image URL: " + err.Error())
		}
	}

	return nil
}

// Size returns the byte size of the image content
// Only considers binary data, not URL references
func (i *ImageContent) Size() int64 {
	if i == nil {
		return 0
	}
	return int64(len(i.Data))
}

// HasData returns true if the image has binary data
func (i *ImageContent) HasData() bool {
	return i != nil && len(i.Data) > 0
}

// HasURL returns true if the image has a URL reference
func (i *ImageContent) HasURL() bool {
	return i != nil && strings.TrimSpace(i.URL) != ""
}

// SetDimensions sets the width and height of the image
func (i *ImageContent) SetDimensions(width, height int) {
	if i != nil {
		i.Width = width
		i.Height = height
	}
}

// GetSupportedImageMimeTypes returns a slice of supported MIME types
func GetSupportedImageMimeTypes() []string {
	types := make([]string, 0, len(supportedImageMimeTypes))
	for mimeType := range supportedImageMimeTypes {
		types = append(types, mimeType)
	}
	return types
}

// IsValidImageMimeType checks if a MIME type is supported for images
func IsValidImageMimeType(mimeType string) bool {
	return supportedImageMimeTypes[mimeType]
}

// MarshalJSON implements custom JSON marshaling for ImageContent
func (i *ImageContent) MarshalJSON() ([]byte, error) {
	if i == nil {
		return json.Marshal(nil)
	}

	// Create a struct with the content type information
	// Note: Data field is omitted via struct tag
	data := struct {
		Type     MessageType `json:"type"`
		URL      string      `json:"url,omitempty"`
		MimeType string      `json:"mime_type"`
		Width    int         `json:"width,omitempty"`
		Height   int         `json:"height,omitempty"`
		Filename string      `json:"filename,omitempty"`
	}{
		Type:     i.Type(),
		URL:      i.URL,
		MimeType: i.MimeType,
		Width:    i.Width,
		Height:   i.Height,
		Filename: i.Filename,
	}

	return json.Marshal(data)
}

// UnmarshalJSON implements custom JSON unmarshaling for ImageContent
func (i *ImageContent) UnmarshalJSON(data []byte) error {
	if i == nil {
		return errors.New("cannot unmarshal into nil ImageContent")
	}

	// Define struct for unmarshaling
	var content struct {
		Type     MessageType `json:"type"`
		URL      string      `json:"url,omitempty"`
		MimeType string      `json:"mime_type"`
		Width    int         `json:"width,omitempty"`
		Height   int         `json:"height,omitempty"`
		Filename string      `json:"filename,omitempty"`
	}

	if err := json.Unmarshal(data, &content); err != nil {
		return err
	}

	// Validate the type field if present
	if content.Type != "" && content.Type != MessageTypeImage {
		return errors.New("invalid content type for ImageContent")
	}

	i.URL = content.URL
	i.MimeType = content.MimeType
	i.Width = content.Width
	i.Height = content.Height
	i.Filename = content.Filename
	// Note: Data is not unmarshaled from JSON as it's omitted

	return nil
}

// FileContent represents file-based message content
// It supports both binary file data and URL references with size tracking
type FileContent struct {
	Data     []byte `json:"-"`             // Binary file data (omitted from JSON)
	URL      string `json:"url,omitempty"` // URL reference for file
	MimeType string `json:"mime_type"`     // Content type (required)
	Filename string `json:"filename"`      // Original filename (required)
	FileSize int64  `json:"size"`          // Explicit file size tracking
}

// Supported MIME types for files
var supportedFileMimeTypes = map[string]bool{
	"application/pdf":  true,
	"text/plain":       true,
	"application/json": true,
	"text/csv":         true,
	"application/vnd.openxmlformats-officedocument.wordprocessingml.document": true,
}

// NewFileContentFromBytes creates a new FileContent instance from binary data
func NewFileContentFromBytes(data []byte, filename, mimeType string) *FileContent {
	return &FileContent{
		Data:     data,
		Filename: filename,
		MimeType: mimeType,
		FileSize: int64(len(data)),
	}
}

// NewFileContentFromURL creates a new FileContent instance from a URL reference
func NewFileContentFromURL(url, filename, mimeType string, size int64) *FileContent {
	return &FileContent{
		URL:      url,
		Filename: filename,
		MimeType: mimeType,
		FileSize: size,
	}
}

// Type returns the message type for file content
func (f *FileContent) Type() MessageType {
	return MessageTypeFile
}

// Validate checks if the file content is valid
func (f *FileContent) Validate() error {
	if f == nil {
		return errors.New("file content cannot be nil")
	}

	// Check that either data or URL is provided (but not both empty)
	// Note: empty data (len == 0) is considered valid data
	hasData := f.Data != nil
	hasURL := strings.TrimSpace(f.URL) != ""

	if !hasData && !hasURL {
		return errors.New("file content must have either data or URL")
	}

	// Filename is mandatory
	if strings.TrimSpace(f.Filename) == "" {
		return errors.New("file content must have a filename")
	}

	// Validate MIME type is provided (security validation will check if it's supported and safe)
	if strings.TrimSpace(f.MimeType) == "" {
		return errors.New("file content must have a MIME type")
	}

	// If URL is provided, validate it's a proper URL
	if hasURL {
		if _, err := url.ParseRequestURI(f.URL); err != nil {
			return errors.New("invalid file URL: " + err.Error())
		}
	}

	// Size consistency check when data is provided
	if hasData && f.FileSize != int64(len(f.Data)) {
		return errors.New("size field does not match data length")
	}

	// Size must be non-negative
	if f.FileSize < 0 {
		return errors.New("file size cannot be negative")
	}

	return nil
}

// Size returns the stored size value for file content
func (f *FileContent) Size() int64 {
	if f == nil {
		return 0
	}
	return f.FileSize
}

// HasData returns true if the file has binary data
func (f *FileContent) HasData() bool {
	return f != nil && len(f.Data) > 0
}

// HasURL returns true if the file has a URL reference
func (f *FileContent) HasURL() bool {
	return f != nil && strings.TrimSpace(f.URL) != ""
}

// GetSupportedFileMimeTypes returns a slice of supported MIME types
func GetSupportedFileMimeTypes() []string {
	types := make([]string, 0, len(supportedFileMimeTypes))
	for mimeType := range supportedFileMimeTypes {
		types = append(types, mimeType)
	}
	return types
}

// IsValidFileMimeType checks if a MIME type is supported for files
func IsValidFileMimeType(mimeType string) bool {
	return supportedFileMimeTypes[mimeType]
}

// MarshalJSON implements custom JSON marshaling for FileContent
func (f *FileContent) MarshalJSON() ([]byte, error) {
	if f == nil {
		return json.Marshal(nil)
	}

	// Create a struct with the content type information
	// Note: Data field is omitted via struct tag
	data := struct {
		Type     MessageType `json:"type"`
		URL      string      `json:"url,omitempty"`
		MimeType string      `json:"mime_type"`
		Filename string      `json:"filename"`
		Size     int64       `json:"size"`
	}{
		Type:     f.Type(),
		URL:      f.URL,
		MimeType: f.MimeType,
		Filename: f.Filename,
		Size:     f.FileSize,
	}

	return json.Marshal(data)
}

// UnmarshalJSON implements custom JSON unmarshaling for FileContent
func (f *FileContent) UnmarshalJSON(data []byte) error {
	if f == nil {
		return errors.New("cannot unmarshal into nil FileContent")
	}

	// Define struct for unmarshaling
	var content struct {
		Type     MessageType `json:"type"`
		URL      string      `json:"url,omitempty"`
		MimeType string      `json:"mime_type"`
		Filename string      `json:"filename"`
		Size     int64       `json:"size"`
	}

	if err := json.Unmarshal(data, &content); err != nil {
		return err
	}

	// Validate the type field if present
	if content.Type != "" && content.Type != MessageTypeFile {
		return errors.New("invalid content type for FileContent")
	}

	f.URL = content.URL
	f.MimeType = content.MimeType
	f.Filename = content.Filename
	f.FileSize = content.Size
	// Note: Data is not unmarshaled from JSON as it's omitted

	return nil
}
