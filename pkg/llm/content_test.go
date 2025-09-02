package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// Test the constructor function
func TestNewTextContent(t *testing.T) {
	tests := []struct {
		name string
		text string
		want string
	}{
		{
			name: "simple text",
			text: "Hello, world!",
			want: "Hello, world!",
		},
		{
			name: "empty text",
			text: "",
			want: "",
		},
		{
			name: "whitespace text",
			text: "   \n\t  ",
			want: "   \n\t  ",
		},
		{
			name: "unicode text",
			text: "Hello, 世界! 🌍",
			want: "Hello, 世界! 🌍",
		},
		{
			name: "multiline text",
			text: "Line 1\nLine 2\nLine 3",
			want: "Line 1\nLine 2\nLine 3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewTextContent(tt.text)
			if got == nil {
				t.Error("NewTextContent returned nil")
				return
			}
			if got.Text != tt.want {
				t.Errorf("NewTextContent() = %q, want %q", got.Text, tt.want)
			}
		})
	}
}

// Test the Type method
func TestTextContent_Type(t *testing.T) {
	tests := []struct {
		name string
		text string
		want MessageType
	}{
		{
			name: "valid text content",
			text: "Hello, world!",
			want: MessageTypeText,
		},
		{
			name: "empty text content",
			text: "",
			want: MessageTypeText,
		},
		{
			name: "unicode text content",
			text: "世界",
			want: MessageTypeText,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tc := NewTextContent(tt.text)
			if got := tc.Type(); got != tt.want {
				t.Errorf("TextContent.Type() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Test nil receiver for Type method
func TestTextContent_Type_NilReceiver(t *testing.T) {
	var tc *TextContent
	got := tc.Type()
	want := MessageTypeText
	if got != want {
		t.Errorf("nil TextContent.Type() = %v, want %v", got, want)
	}
}

// Test the Validate method
func TestTextContent_Validate(t *testing.T) {
	tests := []struct {
		name    string
		content *TextContent
		wantErr bool
		errMsg  string
	}{
		{
			name:    "valid text",
			content: NewTextContent("Hello, world!"),
			wantErr: false,
		},
		{
			name:    "valid unicode text",
			content: NewTextContent("Hello, 世界! 🌍"),
			wantErr: false,
		},
		{
			name:    "valid multiline text",
			content: NewTextContent("Line 1\nLine 2"),
			wantErr: false,
		},
		{
			name:    "empty text",
			content: NewTextContent(""),
			wantErr: true,
			errMsg:  "text content cannot be empty",
		},
		{
			name:    "whitespace only",
			content: NewTextContent("   \n\t  "),
			wantErr: true,
			errMsg:  "text content cannot be only whitespace",
		},
		{
			name:    "single space",
			content: NewTextContent(" "),
			wantErr: true,
			errMsg:  "text content cannot be only whitespace",
		},
		{
			name:    "tabs and newlines",
			content: NewTextContent("\t\n\r"),
			wantErr: true,
			errMsg:  "text content cannot be only whitespace",
		},
		{
			name:    "nil content",
			content: nil,
			wantErr: true,
			errMsg:  "text content cannot be nil",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.content.Validate()
			if tt.wantErr {
				if err == nil {
					t.Error("TextContent.Validate() expected error, got nil")
					return
				}
				if err.Error() != tt.errMsg {
					t.Errorf("TextContent.Validate() error = %q, want %q", err.Error(), tt.errMsg)
				}
			} else {
				if err != nil {
					t.Errorf("TextContent.Validate() unexpected error: %v", err)
				}
			}
		})
	}
}

// Test the Size method
func TestTextContent_Size(t *testing.T) {
	tests := []struct {
		name    string
		content *TextContent
		want    int64
	}{
		{
			name:    "empty text",
			content: NewTextContent(""),
			want:    0,
		},
		{
			name:    "ASCII text",
			content: NewTextContent("Hello"),
			want:    5,
		},
		{
			name:    "unicode text",
			content: NewTextContent("世界"),
			want:    6, // Each character is 3 bytes in UTF-8
		},
		{
			name:    "emoji text",
			content: NewTextContent("🌍"),
			want:    4, // Emoji is 4 bytes in UTF-8
		},
		{
			name:    "mixed text",
			content: NewTextContent("Hello, 世界! 🌍"),
			want:    19, // "Hello, " (7) + "世界" (6) + "! " (2) + "🌍" (4)
		},
		{
			name:    "multiline text",
			content: NewTextContent("Line 1\nLine 2"),
			want:    13,
		},
		{
			name:    "nil content",
			content: nil,
			want:    0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.content.Size()
			if got != tt.want {
				t.Errorf("TextContent.Size() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Test the GetText method
func TestTextContent_GetText(t *testing.T) {
	tests := []struct {
		name    string
		content *TextContent
		want    string
	}{
		{
			name:    "valid text",
			content: NewTextContent("Hello, world!"),
			want:    "Hello, world!",
		},
		{
			name:    "empty text",
			content: NewTextContent(""),
			want:    "",
		},
		{
			name:    "unicode text",
			content: NewTextContent("Hello, 世界! 🌍"),
			want:    "Hello, 世界! 🌍",
		},
		{
			name:    "nil content",
			content: nil,
			want:    "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.content.GetText()
			if got != tt.want {
				t.Errorf("TextContent.GetText() = %q, want %q", got, tt.want)
			}
		})
	}
}

// Test the IsEmpty method
func TestTextContent_IsEmpty(t *testing.T) {
	tests := []struct {
		name    string
		content *TextContent
		want    bool
	}{
		{
			name:    "valid text",
			content: NewTextContent("Hello"),
			want:    false,
		},
		{
			name:    "empty text",
			content: NewTextContent(""),
			want:    true,
		},
		{
			name:    "whitespace only",
			content: NewTextContent("   \n\t  "),
			want:    true,
		},
		{
			name:    "single space",
			content: NewTextContent(" "),
			want:    true,
		},
		{
			name:    "text with leading/trailing whitespace",
			content: NewTextContent("  Hello  "),
			want:    false,
		},
		{
			name:    "nil content",
			content: nil,
			want:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.content.IsEmpty()
			if got != tt.want {
				t.Errorf("TextContent.IsEmpty() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Test JSON marshaling
func TestTextContent_MarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		content *TextContent
		want    string
		wantErr bool
	}{
		{
			name:    "valid text content",
			content: NewTextContent("Hello, world!"),
			want:    `{"type":"text","text":"Hello, world!"}`,
			wantErr: false,
		},
		{
			name:    "empty text content",
			content: NewTextContent(""),
			want:    `{"type":"text","text":""}`,
			wantErr: false,
		},
		{
			name:    "unicode text content",
			content: NewTextContent("Hello, 世界! 🌍"),
			want:    `{"type":"text","text":"Hello, 世界! 🌍"}`,
			wantErr: false,
		},
		{
			name:    "text with special characters",
			content: NewTextContent("Line 1\nLine 2\tTabbed"),
			want:    `{"type":"text","text":"Line 1\nLine 2\tTabbed"}`,
			wantErr: false,
		},
		{
			name:    "nil content",
			content: nil,
			want:    `null`,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.content.MarshalJSON()
			if (err != nil) != tt.wantErr {
				t.Errorf("TextContent.MarshalJSON() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if string(got) != tt.want {
				t.Errorf("TextContent.MarshalJSON() = %s, want %s", string(got), tt.want)
			}
		})
	}
}

// Test JSON unmarshaling
func TestTextContent_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name     string
		jsonData string
		want     *TextContent
		wantErr  bool
		errMsg   string
	}{
		{
			name:     "valid JSON with type",
			jsonData: `{"type":"text","text":"Hello, world!"}`,
			want:     NewTextContent("Hello, world!"),
			wantErr:  false,
		},
		{
			name:     "valid JSON without type",
			jsonData: `{"text":"Hello, world!"}`,
			want:     NewTextContent("Hello, world!"),
			wantErr:  false,
		},
		{
			name:     "empty text",
			jsonData: `{"type":"text","text":""}`,
			want:     NewTextContent(""),
			wantErr:  false,
		},
		{
			name:     "unicode text",
			jsonData: `{"type":"text","text":"Hello, 世界! 🌍"}`,
			want:     NewTextContent("Hello, 世界! 🌍"),
			wantErr:  false,
		},
		{
			name:     "text with special characters",
			jsonData: `{"type":"text","text":"Line 1\nLine 2\tTabbed"}`,
			want:     NewTextContent("Line 1\nLine 2\tTabbed"),
			wantErr:  false,
		},
		{
			name:     "invalid content type",
			jsonData: `{"type":"image","text":"Hello"}`,
			want:     nil,
			wantErr:  true,
			errMsg:   "invalid content type for TextContent",
		},
		{
			name:     "invalid JSON",
			jsonData: `{"type":"text","text":}`,
			want:     nil,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var tc TextContent
			err := tc.UnmarshalJSON([]byte(tt.jsonData))

			if tt.wantErr {
				if err == nil {
					t.Error("TextContent.UnmarshalJSON() expected error, got nil")
					return
				}
				if tt.errMsg != "" && err.Error() != tt.errMsg {
					t.Errorf("TextContent.UnmarshalJSON() error = %q, want %q", err.Error(), tt.errMsg)
				}
				return
			}

			if err != nil {
				t.Errorf("TextContent.UnmarshalJSON() unexpected error: %v", err)
				return
			}

			if tc.Text != tt.want.Text {
				t.Errorf("TextContent.UnmarshalJSON() text = %q, want %q", tc.Text, tt.want.Text)
			}
		})
	}
}

// Test unmarshaling into nil TextContent
func TestTextContent_UnmarshalJSON_NilReceiver(t *testing.T) {
	var tc *TextContent
	err := tc.UnmarshalJSON([]byte(`{"text":"Hello"}`))
	if err == nil {
		t.Error("Expected error when unmarshaling into nil TextContent")
	}
	expectedErr := "cannot unmarshal into nil TextContent"
	if err.Error() != expectedErr {
		t.Errorf("UnmarshalJSON() error = %q, want %q", err.Error(), expectedErr)
	}
}

// Test round-trip JSON marshaling/unmarshaling
func TestTextContent_JSONRoundTrip(t *testing.T) {
	tests := []struct {
		name     string
		original *TextContent
	}{
		{
			name:     "simple text",
			original: NewTextContent("Hello, world!"),
		},
		{
			name:     "empty text",
			original: NewTextContent(""),
		},
		{
			name:     "unicode text",
			original: NewTextContent("Hello, 世界! 🌍"),
		},
		{
			name:     "multiline text",
			original: NewTextContent("Line 1\nLine 2\nLine 3"),
		},
		{
			name:     "text with special characters",
			original: NewTextContent("Special: \t\n\r\"\\"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Marshal to JSON
			data, err := json.Marshal(tt.original)
			if err != nil {
				t.Errorf("Failed to marshal TextContent: %v", err)
				return
			}

			// Unmarshal from JSON
			var restored TextContent
			err = json.Unmarshal(data, &restored)
			if err != nil {
				t.Errorf("Failed to unmarshal TextContent: %v", err)
				return
			}

			// Compare original and restored
			if restored.Text != tt.original.Text {
				t.Errorf("Round-trip failed: got %q, want %q", restored.Text, tt.original.Text)
			}

			// Verify interface compliance
			if restored.Type() != tt.original.Type() {
				t.Errorf("Type mismatch after round-trip: got %v, want %v", restored.Type(), tt.original.Type())
			}
		})
	}
}

// Test that TextContent implements MessageContent interface
func TestTextContent_ImplementsMessageContent(t *testing.T) {
	var _ MessageContent = (*TextContent)(nil)
	var _ MessageContent = NewTextContent("test")
}

// Benchmark tests for performance
func BenchmarkTextContent_Size(b *testing.B) {
	tests := []struct {
		name string
		text string
	}{
		{"short", "Hello"},
		{"medium", strings.Repeat("Hello, world! ", 100)},
		{"long", strings.Repeat("Hello, world! ", 10000)},
		{"unicode", strings.Repeat("Hello, 世界! 🌍 ", 1000)},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			tc := NewTextContent(tt.text)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = tc.Size()
			}
		})
	}
}

func BenchmarkTextContent_Validate(b *testing.B) {
	tc := NewTextContent("Hello, world!")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tc.Validate()
	}
}

func BenchmarkTextContent_JSON(b *testing.B) {
	tc := NewTextContent("Hello, world! This is a test message for JSON benchmarking.")

	b.Run("marshal", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(tc)
		}
	})

	data, _ := json.Marshal(tc)
	b.Run("unmarshal", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var restored TextContent
			_ = json.Unmarshal(data, &restored)
		}
	})
}

// Sample image data for testing
var (
	sampleJPEGData = []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46} // Simple JPEG header
	samplePNGData  = []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}             // PNG signature
)

// Test constructor functions for ImageContent
func TestNewImageContentFromBytes(t *testing.T) {
	tests := []struct {
		name     string
		data     []byte
		mimeType string
		want     string
	}{
		{
			name:     "valid JPEG data",
			data:     sampleJPEGData,
			mimeType: "image/jpeg",
			want:     "image/jpeg",
		},
		{
			name:     "valid PNG data",
			data:     samplePNGData,
			mimeType: "image/png",
			want:     "image/png",
		},
		{
			name:     "empty data",
			data:     []byte{},
			mimeType: "image/jpeg",
			want:     "image/jpeg",
		},
		{
			name:     "nil data",
			data:     nil,
			mimeType: "image/jpeg",
			want:     "image/jpeg",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewImageContentFromBytes(tt.data, tt.mimeType)
			if got == nil {
				t.Error("NewImageContentFromBytes returned nil")
				return
			}
			if got.MimeType != tt.want {
				t.Errorf("MimeType = %v, want %v", got.MimeType, tt.want)
			}
			if len(got.Data) != len(tt.data) {
				t.Errorf("Data length = %d, want %d", len(got.Data), len(tt.data))
			}
			if got.Type() != MessageTypeImage {
				t.Errorf("Type() = %v, want %v", got.Type(), MessageTypeImage)
			}
		})
	}
}

func TestNewImageContentFromURL(t *testing.T) {
	tests := []struct {
		name     string
		url      string
		mimeType string
		wantType string
	}{
		{
			name:     "valid HTTPS URL",
			url:      "https://example.com/image.jpg",
			mimeType: "image/jpeg",
			wantType: "image/jpeg",
		},
		{
			name:     "valid HTTP URL",
			url:      "http://example.com/image.png",
			mimeType: "image/png",
			wantType: "image/png",
		},
		{
			name:     "empty URL",
			url:      "",
			mimeType: "image/jpeg",
			wantType: "image/jpeg",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewImageContentFromURL(tt.url, tt.mimeType)
			if got == nil {
				t.Error("NewImageContentFromURL returned nil")
				return
			}
			if got.URL != tt.url {
				t.Errorf("URL = %v, want %v", got.URL, tt.url)
			}
			if got.MimeType != tt.wantType {
				t.Errorf("MimeType = %v, want %v", got.MimeType, tt.wantType)
			}
			if got.Type() != MessageTypeImage {
				t.Errorf("Type() = %v, want %v", got.Type(), MessageTypeImage)
			}
		})
	}
}

// Test ImageContent interface methods
func TestImageContentType(t *testing.T) {
	content := NewImageContentFromBytes(sampleJPEGData, "image/jpeg")
	if content.Type() != MessageTypeImage {
		t.Errorf("Type() = %v, want %v", content.Type(), MessageTypeImage)
	}
}

func TestImageContentValidate(t *testing.T) {
	tests := []struct {
		name     string
		content  *ImageContent
		wantErr  bool
		errorMsg string
	}{
		{
			name:    "nil content",
			content: nil,
			wantErr: true,
		},
		{
			name: "valid data content",
			content: &ImageContent{
				Data:     sampleJPEGData,
				MimeType: "image/jpeg",
			},
			wantErr: false,
		},
		{
			name: "valid URL content",
			content: &ImageContent{
				URL:      "https://example.com/image.jpg",
				MimeType: "image/jpeg",
			},
			wantErr: false,
		},
		{
			name: "no data or URL",
			content: &ImageContent{
				MimeType: "image/jpeg",
			},
			wantErr:  true,
			errorMsg: "must have either data or URL",
		},
		{
			name: "empty mime type",
			content: &ImageContent{
				Data:     sampleJPEGData,
				MimeType: "",
			},
			wantErr:  true,
			errorMsg: "must have a MIME type",
		},
		{
			name: "whitespace mime type",
			content: &ImageContent{
				Data:     sampleJPEGData,
				MimeType: "   ",
			},
			wantErr:  true,
			errorMsg: "must have a MIME type",
		},
		{
			name: "BMP mime type (now handled by security layer)",
			content: &ImageContent{
				Data:     sampleJPEGData,
				MimeType: "image/bmp",
			},
			wantErr: false, // Basic validation no longer checks MIME type support
		},
		{
			name: "invalid URL",
			content: &ImageContent{
				URL:      "not-a-url",
				MimeType: "image/jpeg",
			},
			wantErr:  true,
			errorMsg: "invalid image URL",
		},
		{
			name: "valid data with supported formats",
			content: &ImageContent{
				Data:     samplePNGData,
				MimeType: "image/png",
			},
			wantErr: false,
		},
		{
			name: "webp format",
			content: &ImageContent{
				Data:     []byte{0x52, 0x49, 0x46, 0x46},
				MimeType: "image/webp",
			},
			wantErr: false,
		},
		{
			name: "gif format",
			content: &ImageContent{
				Data:     []byte{0x47, 0x49, 0x46, 0x38},
				MimeType: "image/gif",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.content.Validate()
			if tt.wantErr {
				if err == nil {
					t.Error("Validate() expected error but got none")
				} else if tt.errorMsg != "" && !strings.Contains(err.Error(), tt.errorMsg) {
					t.Errorf("Validate() error = %v, want error containing %v", err, tt.errorMsg)
				}
			} else {
				if err != nil {
					t.Errorf("Validate() unexpected error = %v", err)
				}
			}
		})
	}
}

func TestImageContentSize(t *testing.T) {
	tests := []struct {
		name    string
		content *ImageContent
		want    int64
	}{
		{
			name:    "nil content",
			content: nil,
			want:    0,
		},
		{
			name: "empty data",
			content: &ImageContent{
				Data:     []byte{},
				MimeType: "image/jpeg",
			},
			want: 0,
		},
		{
			name: "sample data",
			content: &ImageContent{
				Data:     sampleJPEGData,
				MimeType: "image/jpeg",
			},
			want: int64(len(sampleJPEGData)),
		},
		{
			name: "URL only (no data)",
			content: &ImageContent{
				URL:      "https://example.com/image.jpg",
				MimeType: "image/jpeg",
			},
			want: 0,
		},
		{
			name: "large data",
			content: &ImageContent{
				Data:     make([]byte, 1024),
				MimeType: "image/jpeg",
			},
			want: 1024,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.content.Size()
			if got != tt.want {
				t.Errorf("Size() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Test helper methods
func TestImageContentHelperMethods(t *testing.T) {
	t.Run("HasData", func(t *testing.T) {
		tests := []struct {
			name    string
			content *ImageContent
			want    bool
		}{
			{
				name:    "nil content",
				content: nil,
				want:    false,
			},
			{
				name: "with data",
				content: &ImageContent{
					Data:     sampleJPEGData,
					MimeType: "image/jpeg",
				},
				want: true,
			},
			{
				name: "empty data",
				content: &ImageContent{
					Data:     []byte{},
					MimeType: "image/jpeg",
				},
				want: false,
			},
			{
				name: "no data",
				content: &ImageContent{
					URL:      "https://example.com/image.jpg",
					MimeType: "image/jpeg",
				},
				want: false,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got := tt.content.HasData()
				if got != tt.want {
					t.Errorf("HasData() = %v, want %v", got, tt.want)
				}
			})
		}
	})

	t.Run("HasURL", func(t *testing.T) {
		tests := []struct {
			name    string
			content *ImageContent
			want    bool
		}{
			{
				name:    "nil content",
				content: nil,
				want:    false,
			},
			{
				name: "with URL",
				content: &ImageContent{
					URL:      "https://example.com/image.jpg",
					MimeType: "image/jpeg",
				},
				want: true,
			},
			{
				name: "empty URL",
				content: &ImageContent{
					Data:     sampleJPEGData,
					URL:      "",
					MimeType: "image/jpeg",
				},
				want: false,
			},
			{
				name: "whitespace URL",
				content: &ImageContent{
					URL:      "   \t\n  ",
					MimeType: "image/jpeg",
				},
				want: false,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got := tt.content.HasURL()
				if got != tt.want {
					t.Errorf("HasURL() = %v, want %v", got, tt.want)
				}
			})
		}
	})

	t.Run("SetDimensions", func(t *testing.T) {
		content := NewImageContentFromBytes(sampleJPEGData, "image/jpeg")
		content.SetDimensions(1920, 1080)

		if content.Width != 1920 {
			t.Errorf("Width = %d, want 1920", content.Width)
		}
		if content.Height != 1080 {
			t.Errorf("Height = %d, want 1080", content.Height)
		}

		// Test with nil
		var nilContent *ImageContent
		nilContent.SetDimensions(100, 100) // Should not panic
	})
}

// Test MIME type helper functions
func TestImageMimeTypeHelpers(t *testing.T) {
	t.Run("GetSupportedImageMimeTypes", func(t *testing.T) {
		types := GetSupportedImageMimeTypes()
		expectedTypes := []string{"image/jpeg", "image/png", "image/gif", "image/webp"}

		if len(types) != len(expectedTypes) {
			t.Errorf("GetSupportedImageMimeTypes() returned %d types, want %d", len(types), len(expectedTypes))
		}

		// Check that all expected types are present
		typeSet := make(map[string]bool)
		for _, mimeType := range types {
			typeSet[mimeType] = true
		}

		for _, expected := range expectedTypes {
			if !typeSet[expected] {
				t.Errorf("GetSupportedImageMimeTypes() missing %s", expected)
			}
		}
	})

	t.Run("IsValidImageMimeType", func(t *testing.T) {
		tests := []struct {
			mimeType string
			want     bool
		}{
			{"image/jpeg", true},
			{"image/png", true},
			{"image/gif", true},
			{"image/webp", true},
			{"image/bmp", false},
			{"image/tiff", false},
			{"text/plain", false},
			{"", false},
		}

		for _, tt := range tests {
			t.Run(tt.mimeType, func(t *testing.T) {
				got := IsValidImageMimeType(tt.mimeType)
				if got != tt.want {
					t.Errorf("IsValidImageMimeType(%s) = %v, want %v", tt.mimeType, got, tt.want)
				}
			})
		}
	})
}

// Test JSON marshaling/unmarshaling
func TestImageContentJSON(t *testing.T) {
	t.Run("MarshalJSON", func(t *testing.T) {
		tests := []struct {
			name     string
			content  *ImageContent
			wantJSON string
		}{
			{
				name:     "nil content",
				content:  nil,
				wantJSON: "null",
			},
			{
				name: "URL content",
				content: &ImageContent{
					URL:      "https://example.com/image.jpg",
					MimeType: "image/jpeg",
					Width:    1920,
					Height:   1080,
					Filename: "image.jpg",
				},
				wantJSON: `{"type":"image","url":"https://example.com/image.jpg","mime_type":"image/jpeg","width":1920,"height":1080,"filename":"image.jpg"}`,
			},
			{
				name: "data content (data omitted)",
				content: &ImageContent{
					Data:     sampleJPEGData,
					MimeType: "image/jpeg",
				},
				wantJSON: `{"type":"image","mime_type":"image/jpeg"}`,
			},
			{
				name: "minimal content",
				content: &ImageContent{
					URL:      "https://example.com/image.png",
					MimeType: "image/png",
				},
				wantJSON: `{"type":"image","url":"https://example.com/image.png","mime_type":"image/png"}`,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				data, err := json.Marshal(tt.content)
				if err != nil {
					t.Errorf("MarshalJSON() error = %v", err)
					return
				}
				got := string(data)
				if got != tt.wantJSON {
					t.Errorf("MarshalJSON() = %s, want %s", got, tt.wantJSON)
				}
			})
		}
	})

	t.Run("UnmarshalJSON", func(t *testing.T) {
		tests := []struct {
			name    string
			json    string
			wantErr bool
		}{
			{
				name: "valid content",
				json: `{"type":"image","url":"https://example.com/image.jpg","mime_type":"image/jpeg","width":1920,"height":1080,"filename":"image.jpg"}`,
			},
			{
				name: "minimal content",
				json: `{"mime_type":"image/png"}`,
			},
			{
				name:    "invalid type",
				json:    `{"type":"text","mime_type":"image/jpeg"}`,
				wantErr: true,
			},
			{
				name:    "invalid JSON",
				json:    `{invalid}`,
				wantErr: true,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var content ImageContent
				err := json.Unmarshal([]byte(tt.json), &content)

				if tt.wantErr {
					if err == nil {
						t.Error("UnmarshalJSON() expected error but got none")
					}
				} else {
					if err != nil {
						t.Errorf("UnmarshalJSON() unexpected error = %v", err)
					}
				}
			})
		}
	})

	t.Run("round trip", func(t *testing.T) {
		original := &ImageContent{
			URL:      "https://example.com/image.jpg",
			MimeType: "image/jpeg",
			Width:    1920,
			Height:   1080,
			Filename: "test.jpg",
		}

		data, err := json.Marshal(original)
		if err != nil {
			t.Fatalf("Marshal error: %v", err)
		}

		var restored ImageContent
		err = json.Unmarshal(data, &restored)
		if err != nil {
			t.Fatalf("Unmarshal error: %v", err)
		}

		if restored.URL != original.URL {
			t.Errorf("URL = %s, want %s", restored.URL, original.URL)
		}
		if restored.MimeType != original.MimeType {
			t.Errorf("MimeType = %s, want %s", restored.MimeType, original.MimeType)
		}
		if restored.Width != original.Width {
			t.Errorf("Width = %d, want %d", restored.Width, original.Width)
		}
		if restored.Height != original.Height {
			t.Errorf("Height = %d, want %d", restored.Height, original.Height)
		}
		if restored.Filename != original.Filename {
			t.Errorf("Filename = %s, want %s", restored.Filename, original.Filename)
		}
		// Note: Data should not be restored from JSON as it's omitted
		if len(restored.Data) != 0 {
			t.Errorf("Data should be empty after JSON round trip, got %d bytes", len(restored.Data))
		}
	})

	t.Run("nil pointer unmarshal", func(t *testing.T) {
		var content *ImageContent
		err := json.Unmarshal([]byte(`{"mime_type":"image/jpeg"}`), content)
		if err == nil {
			t.Error("UnmarshalJSON() into nil pointer should return error")
		}
	})
}

// Sample file data for testing
var (
	samplePDFData  = []byte{0x25, 0x50, 0x44, 0x46, 0x2D} // Simple PDF header
	sampleTextData = []byte("Hello, world!")
	sampleJSONData = []byte(`{"test": "data"}`)
	sampleCSVData  = []byte("name,value\ntest,123")
)

// Test constructor functions for FileContent
func TestNewFileContentFromBytes(t *testing.T) {
	tests := []struct {
		name     string
		data     []byte
		filename string
		mimeType string
		want     string
	}{
		{
			name:     "valid PDF data",
			data:     samplePDFData,
			filename: "document.pdf",
			mimeType: "application/pdf",
			want:     "application/pdf",
		},
		{
			name:     "valid text data",
			data:     sampleTextData,
			filename: "test.txt",
			mimeType: "text/plain",
			want:     "text/plain",
		},
		{
			name:     "valid JSON data",
			data:     sampleJSONData,
			filename: "data.json",
			mimeType: "application/json",
			want:     "application/json",
		},
		{
			name:     "valid CSV data",
			data:     sampleCSVData,
			filename: "data.csv",
			mimeType: "text/csv",
			want:     "text/csv",
		},
		{
			name:     "empty data",
			data:     []byte{},
			filename: "empty.txt",
			mimeType: "text/plain",
			want:     "text/plain",
		},
		{
			name:     "nil data",
			data:     nil,
			filename: "nil.txt",
			mimeType: "text/plain",
			want:     "text/plain",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewFileContentFromBytes(tt.data, tt.filename, tt.mimeType)
			if got == nil {
				t.Error("NewFileContentFromBytes returned nil")
				return
			}
			if got.MimeType != tt.want {
				t.Errorf("MimeType = %v, want %v", got.MimeType, tt.want)
			}
			if got.Filename != tt.filename {
				t.Errorf("Filename = %v, want %v", got.Filename, tt.filename)
			}
			if len(got.Data) != len(tt.data) {
				t.Errorf("Data length = %d, want %d", len(got.Data), len(tt.data))
			}
			if got.FileSize != int64(len(tt.data)) {
				t.Errorf("FileSize = %d, want %d", got.FileSize, len(tt.data))
			}
			if got.Type() != MessageTypeFile {
				t.Errorf("Type() = %v, want %v", got.Type(), MessageTypeFile)
			}
		})
	}
}

func TestNewFileContentFromURL(t *testing.T) {
	tests := []struct {
		name     string
		url      string
		filename string
		mimeType string
		size     int64
	}{
		{
			name:     "valid HTTPS URL",
			url:      "https://example.com/document.pdf",
			filename: "document.pdf",
			mimeType: "application/pdf",
			size:     1024,
		},
		{
			name:     "valid HTTP URL",
			url:      "http://example.com/data.csv",
			filename: "data.csv",
			mimeType: "text/csv",
			size:     2048,
		},
		{
			name:     "empty URL",
			url:      "",
			filename: "test.txt",
			mimeType: "text/plain",
			size:     0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewFileContentFromURL(tt.url, tt.filename, tt.mimeType, tt.size)
			if got == nil {
				t.Error("NewFileContentFromURL returned nil")
				return
			}
			if got.URL != tt.url {
				t.Errorf("URL = %v, want %v", got.URL, tt.url)
			}
			if got.Filename != tt.filename {
				t.Errorf("Filename = %v, want %v", got.Filename, tt.filename)
			}
			if got.MimeType != tt.mimeType {
				t.Errorf("MimeType = %v, want %v", got.MimeType, tt.mimeType)
			}
			if got.FileSize != tt.size {
				t.Errorf("FileSize = %v, want %v", got.FileSize, tt.size)
			}
			if got.Type() != MessageTypeFile {
				t.Errorf("Type() = %v, want %v", got.Type(), MessageTypeFile)
			}
		})
	}
}

// Test FileContent interface methods
func TestFileContentType(t *testing.T) {
	content := NewFileContentFromBytes(samplePDFData, "test.pdf", "application/pdf")
	if content.Type() != MessageTypeFile {
		t.Errorf("Type() = %v, want %v", content.Type(), MessageTypeFile)
	}
}

func TestFileContentValidate(t *testing.T) {
	tests := []struct {
		name     string
		content  *FileContent
		wantErr  bool
		errorMsg string
	}{
		{
			name:    "nil content",
			content: nil,
			wantErr: true,
		},
		{
			name: "valid data content",
			content: &FileContent{
				Data:     samplePDFData,
				Filename: "document.pdf",
				MimeType: "application/pdf",
				FileSize: int64(len(samplePDFData)),
			},
			wantErr: false,
		},
		{
			name: "valid URL content",
			content: &FileContent{
				URL:      "https://example.com/document.pdf",
				Filename: "document.pdf",
				MimeType: "application/pdf",
				FileSize: 1024,
			},
			wantErr: false,
		},
		{
			name: "no data or URL",
			content: &FileContent{
				Filename: "document.pdf",
				MimeType: "application/pdf",
				FileSize: 0,
			},
			wantErr:  true,
			errorMsg: "must have either data or URL",
		},
		{
			name: "empty filename",
			content: &FileContent{
				Data:     samplePDFData,
				Filename: "",
				MimeType: "application/pdf",
				FileSize: int64(len(samplePDFData)),
			},
			wantErr:  true,
			errorMsg: "must have a filename",
		},
		{
			name: "whitespace filename",
			content: &FileContent{
				Data:     samplePDFData,
				Filename: "   ",
				MimeType: "application/pdf",
				FileSize: int64(len(samplePDFData)),
			},
			wantErr:  true,
			errorMsg: "must have a filename",
		},
		{
			name: "filename with path traversal (now handled by security layer)",
			content: &FileContent{
				Data:     samplePDFData,
				Filename: "../test.pdf",
				MimeType: "application/pdf",
				FileSize: int64(len(samplePDFData)),
			},
			wantErr: false, // Basic validation no longer checks for path characters
		},
		{
			name: "filename with forward slash (now handled by security layer)",
			content: &FileContent{
				Data:     samplePDFData,
				Filename: "path/test.pdf",
				MimeType: "application/pdf",
				FileSize: int64(len(samplePDFData)),
			},
			wantErr: false, // Basic validation no longer checks for path characters
		},
		{
			name: "filename with backslash (now handled by security layer)",
			content: &FileContent{
				Data:     samplePDFData,
				Filename: "path\\test.pdf",
				MimeType: "application/pdf",
				FileSize: int64(len(samplePDFData)),
			},
			wantErr: false, // Basic validation no longer checks for path characters
		},
		{
			name: "empty mime type",
			content: &FileContent{
				Data:     samplePDFData,
				Filename: "document.pdf",
				MimeType: "",
				FileSize: int64(len(samplePDFData)),
			},
			wantErr:  true,
			errorMsg: "must have a MIME type",
		},
		{
			name: "whitespace mime type",
			content: &FileContent{
				Data:     samplePDFData,
				Filename: "document.pdf",
				MimeType: "   ",
				FileSize: int64(len(samplePDFData)),
			},
			wantErr:  true,
			errorMsg: "must have a MIME type",
		},
		{
			name: "executable mime type (now handled by security layer)",
			content: &FileContent{
				Data:     samplePDFData,
				Filename: "document.exe",
				MimeType: "application/x-msdownload",
				FileSize: int64(len(samplePDFData)),
			},
			wantErr: false, // Basic validation no longer checks MIME type support
		},
		{
			name: "invalid URL",
			content: &FileContent{
				URL:      "not-a-url",
				Filename: "document.pdf",
				MimeType: "application/pdf",
				FileSize: 1024,
			},
			wantErr:  true,
			errorMsg: "invalid file URL",
		},
		{
			name: "size mismatch with data",
			content: &FileContent{
				Data:     samplePDFData,
				Filename: "document.pdf",
				MimeType: "application/pdf",
				FileSize: 999, // Wrong size
			},
			wantErr:  true,
			errorMsg: "size field does not match data length",
		},
		{
			name: "negative size",
			content: &FileContent{
				URL:      "https://example.com/document.pdf",
				Filename: "document.pdf",
				MimeType: "application/pdf",
				FileSize: -1,
			},
			wantErr:  true,
			errorMsg: "file size cannot be negative",
		},
		{
			name: "supported formats",
			content: &FileContent{
				Data:     sampleTextData,
				Filename: "test.txt",
				MimeType: "text/plain",
				FileSize: int64(len(sampleTextData)),
			},
			wantErr: false,
		},
		{
			name: "Word document format",
			content: &FileContent{
				Data:     []byte("PK\x03\x04"),
				Filename: "document.docx",
				MimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
				FileSize: 4,
			},
			wantErr: false,
		},
		{
			name: "zero size file with no data",
			content: &FileContent{
				URL:      "https://example.com/empty.txt",
				Filename: "empty.txt",
				MimeType: "text/plain",
				FileSize: 0,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.content.Validate()
			if tt.wantErr {
				if err == nil {
					t.Error("Validate() expected error but got none")
				} else if tt.errorMsg != "" && !strings.Contains(err.Error(), tt.errorMsg) {
					t.Errorf("Validate() error = %v, want error containing %v", err, tt.errorMsg)
				}
			} else {
				if err != nil {
					t.Errorf("Validate() unexpected error = %v", err)
				}
			}
		})
	}
}

func TestFileContentSize(t *testing.T) {
	tests := []struct {
		name    string
		content *FileContent
		want    int64
	}{
		{
			name:    "nil content",
			content: nil,
			want:    0,
		},
		{
			name: "empty data",
			content: &FileContent{
				Data:     []byte{},
				Filename: "empty.txt",
				MimeType: "text/plain",
				FileSize: 0,
			},
			want: 0,
		},
		{
			name: "sample data",
			content: &FileContent{
				Data:     samplePDFData,
				Filename: "document.pdf",
				MimeType: "application/pdf",
				FileSize: int64(len(samplePDFData)),
			},
			want: int64(len(samplePDFData)),
		},
		{
			name: "URL only (explicit size)",
			content: &FileContent{
				URL:      "https://example.com/document.pdf",
				Filename: "document.pdf",
				MimeType: "application/pdf",
				FileSize: 2048,
			},
			want: 2048,
		},
		{
			name: "large file size",
			content: &FileContent{
				URL:      "https://example.com/large.pdf",
				Filename: "large.pdf",
				MimeType: "application/pdf",
				FileSize: 1024 * 1024, // 1MB
			},
			want: 1024 * 1024,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.content.Size()
			if got != tt.want {
				t.Errorf("Size() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Test helper methods
func TestFileContentHelperMethods(t *testing.T) {
	t.Run("HasData", func(t *testing.T) {
		tests := []struct {
			name    string
			content *FileContent
			want    bool
		}{
			{
				name:    "nil content",
				content: nil,
				want:    false,
			},
			{
				name: "with data",
				content: &FileContent{
					Data:     samplePDFData,
					Filename: "document.pdf",
					MimeType: "application/pdf",
					FileSize: int64(len(samplePDFData)),
				},
				want: true,
			},
			{
				name: "empty data",
				content: &FileContent{
					Data:     []byte{},
					Filename: "empty.pdf",
					MimeType: "application/pdf",
					FileSize: 0,
				},
				want: false,
			},
			{
				name: "no data",
				content: &FileContent{
					URL:      "https://example.com/document.pdf",
					Filename: "document.pdf",
					MimeType: "application/pdf",
					FileSize: 1024,
				},
				want: false,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got := tt.content.HasData()
				if got != tt.want {
					t.Errorf("HasData() = %v, want %v", got, tt.want)
				}
			})
		}
	})

	t.Run("HasURL", func(t *testing.T) {
		tests := []struct {
			name    string
			content *FileContent
			want    bool
		}{
			{
				name:    "nil content",
				content: nil,
				want:    false,
			},
			{
				name: "with URL",
				content: &FileContent{
					URL:      "https://example.com/document.pdf",
					Filename: "document.pdf",
					MimeType: "application/pdf",
					FileSize: 1024,
				},
				want: true,
			},
			{
				name: "empty URL",
				content: &FileContent{
					Data:     samplePDFData,
					URL:      "",
					Filename: "document.pdf",
					MimeType: "application/pdf",
					FileSize: int64(len(samplePDFData)),
				},
				want: false,
			},
			{
				name: "whitespace URL",
				content: &FileContent{
					URL:      "   \t\n  ",
					Filename: "document.pdf",
					MimeType: "application/pdf",
					FileSize: 0,
				},
				want: false,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got := tt.content.HasURL()
				if got != tt.want {
					t.Errorf("HasURL() = %v, want %v", got, tt.want)
				}
			})
		}
	})
}

// Test MIME type helper functions
func TestFileMimeTypeHelpers(t *testing.T) {
	t.Run("GetSupportedFileMimeTypes", func(t *testing.T) {
		types := GetSupportedFileMimeTypes()
		expectedTypes := []string{
			"application/pdf",
			"text/plain",
			"application/json",
			"text/csv",
			"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
		}

		if len(types) != len(expectedTypes) {
			t.Errorf("GetSupportedFileMimeTypes() returned %d types, want %d", len(types), len(expectedTypes))
		}

		// Check that all expected types are present
		typeSet := make(map[string]bool)
		for _, mimeType := range types {
			typeSet[mimeType] = true
		}

		for _, expected := range expectedTypes {
			if !typeSet[expected] {
				t.Errorf("GetSupportedFileMimeTypes() missing %s", expected)
			}
		}
	})

	t.Run("IsValidFileMimeType", func(t *testing.T) {
		tests := []struct {
			mimeType string
			want     bool
		}{
			{"application/pdf", true},
			{"text/plain", true},
			{"application/json", true},
			{"text/csv", true},
			{"application/vnd.openxmlformats-officedocument.wordprocessingml.document", true},
			{"application/msword", false},
			{"image/jpeg", false},
			{"text/html", false},
			{"", false},
		}

		for _, tt := range tests {
			t.Run(tt.mimeType, func(t *testing.T) {
				got := IsValidFileMimeType(tt.mimeType)
				if got != tt.want {
					t.Errorf("IsValidFileMimeType(%s) = %v, want %v", tt.mimeType, got, tt.want)
				}
			})
		}
	})
}

// Test JSON marshaling/unmarshaling
func TestFileContentJSON(t *testing.T) {
	t.Run("MarshalJSON", func(t *testing.T) {
		tests := []struct {
			name     string
			content  *FileContent
			wantJSON string
		}{
			{
				name:     "nil content",
				content:  nil,
				wantJSON: "null",
			},
			{
				name: "URL content",
				content: &FileContent{
					URL:      "https://example.com/document.pdf",
					Filename: "document.pdf",
					MimeType: "application/pdf",
					FileSize: 1024,
				},
				wantJSON: `{"type":"file","url":"https://example.com/document.pdf","mime_type":"application/pdf","filename":"document.pdf","size":1024}`,
			},
			{
				name: "data content (data omitted)",
				content: &FileContent{
					Data:     samplePDFData,
					Filename: "document.pdf",
					MimeType: "application/pdf",
					FileSize: int64(len(samplePDFData)),
				},
				wantJSON: `{"type":"file","mime_type":"application/pdf","filename":"document.pdf","size":5}`,
			},
			{
				name: "minimal content",
				content: &FileContent{
					URL:      "https://example.com/data.txt",
					Filename: "data.txt",
					MimeType: "text/plain",
					FileSize: 0,
				},
				wantJSON: `{"type":"file","url":"https://example.com/data.txt","mime_type":"text/plain","filename":"data.txt","size":0}`,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				data, err := json.Marshal(tt.content)
				if err != nil {
					t.Errorf("MarshalJSON() error = %v", err)
					return
				}
				got := string(data)
				if got != tt.wantJSON {
					t.Errorf("MarshalJSON() = %s, want %s", got, tt.wantJSON)
				}
			})
		}
	})

	t.Run("UnmarshalJSON", func(t *testing.T) {
		tests := []struct {
			name    string
			json    string
			wantErr bool
		}{
			{
				name: "valid content",
				json: `{"type":"file","url":"https://example.com/document.pdf","mime_type":"application/pdf","filename":"document.pdf","size":1024}`,
			},
			{
				name: "minimal content",
				json: `{"mime_type":"text/plain","filename":"test.txt","size":0}`,
			},
			{
				name:    "invalid type",
				json:    `{"type":"image","mime_type":"application/pdf","filename":"test.pdf","size":1024}`,
				wantErr: true,
			},
			{
				name:    "invalid JSON",
				json:    `{invalid}`,
				wantErr: true,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var content FileContent
				err := json.Unmarshal([]byte(tt.json), &content)

				if tt.wantErr {
					if err == nil {
						t.Error("UnmarshalJSON() expected error but got none")
					}
				} else {
					if err != nil {
						t.Errorf("UnmarshalJSON() unexpected error = %v", err)
					}
				}
			})
		}
	})

	t.Run("round trip", func(t *testing.T) {
		original := &FileContent{
			URL:      "https://example.com/document.pdf",
			Filename: "test.pdf",
			MimeType: "application/pdf",
			FileSize: 2048,
		}

		data, err := json.Marshal(original)
		if err != nil {
			t.Fatalf("Marshal error: %v", err)
		}

		var restored FileContent
		err = json.Unmarshal(data, &restored)
		if err != nil {
			t.Fatalf("Unmarshal error: %v", err)
		}

		if restored.URL != original.URL {
			t.Errorf("URL = %s, want %s", restored.URL, original.URL)
		}
		if restored.Filename != original.Filename {
			t.Errorf("Filename = %s, want %s", restored.Filename, original.Filename)
		}
		if restored.MimeType != original.MimeType {
			t.Errorf("MimeType = %s, want %s", restored.MimeType, original.MimeType)
		}
		if restored.FileSize != original.FileSize {
			t.Errorf("FileSize = %d, want %d", restored.FileSize, original.FileSize)
		}
		// Note: Data should not be restored from JSON as it's omitted
		if len(restored.Data) != 0 {
			t.Errorf("Data should be empty after JSON round trip, got %d bytes", len(restored.Data))
		}
	})

	t.Run("nil pointer unmarshal", func(t *testing.T) {
		var content *FileContent
		err := json.Unmarshal([]byte(`{"mime_type":"application/pdf","filename":"test.pdf","size":1024}`), content)
		if err == nil {
			t.Error("UnmarshalJSON() into nil pointer should return error")
		}
	})
}

// Test edge cases and special scenarios
func TestFileContentEdgeCases(t *testing.T) {
	t.Run("very long filename", func(t *testing.T) {
		longFilename := strings.Repeat("a", 255) + ".txt"
		content := NewFileContentFromBytes(sampleTextData, longFilename, "text/plain")
		err := content.Validate()
		if err != nil {
			t.Errorf("Validate() with long filename should not error, got: %v", err)
		}
	})

	t.Run("zero size file with data", func(t *testing.T) {
		content := NewFileContentFromBytes([]byte{}, "empty.txt", "text/plain")
		if content.Size() != 0 {
			t.Errorf("Size() = %d, want 0 for empty file", content.Size())
		}
		if err := content.Validate(); err != nil {
			t.Errorf("Validate() should not error for empty file, got: %v", err)
		}
	})

	t.Run("file extension and MIME type consistency", func(t *testing.T) {
		// This tests that we can have mismatched extensions and MIME types
		// (the validation only checks MIME type, not extension consistency)
		content := NewFileContentFromBytes(samplePDFData, "document.txt", "application/pdf")
		err := content.Validate()
		if err != nil {
			t.Errorf("Validate() should allow mismatched extension and MIME type, got: %v", err)
		}
	})
}

// Test that FileContent implements MessageContent interface
func TestFileContent_ImplementsMessageContent(t *testing.T) {
	var _ MessageContent = (*FileContent)(nil)
	var _ MessageContent = NewFileContentFromBytes(samplePDFData, "test.pdf", "application/pdf")
}

// Benchmark tests for performance
func BenchmarkFileContent_Size(b *testing.B) {
	tests := []struct {
		name string
		size int64
	}{
		{"small", 1024},
		{"medium", 1024 * 1024},
		{"large", 100 * 1024 * 1024},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			fc := NewFileContentFromURL("https://example.com/test.pdf", "test.pdf", "application/pdf", tt.size)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = fc.Size()
			}
		})
	}
}

func BenchmarkFileContent_Validate(b *testing.B) {
	fc := NewFileContentFromBytes(samplePDFData, "test.pdf", "application/pdf")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fc.Validate()
	}
}

func BenchmarkFileContent_JSON(b *testing.B) {
	fc := NewFileContentFromURL("https://example.com/test.pdf", "test.pdf", "application/pdf", 2048)

	b.Run("marshal", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(fc)
		}
	})

	data, _ := json.Marshal(fc)
	b.Run("unmarshal", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var restored FileContent
			_ = json.Unmarshal(data, &restored)
		}
	})
}
