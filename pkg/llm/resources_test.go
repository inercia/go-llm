package llm

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// Mock content types for testing
type mockContent struct {
	contentType MessageType
	size        int64
	data        []byte
	valid       bool
}

func (m *mockContent) Type() MessageType {
	return m.contentType
}

func (m *mockContent) Size() int64 {
	return m.size
}

func (m *mockContent) Validate() error {
	if !m.valid {
		return fmt.Errorf("mock content validation failed")
	}
	return nil
}

func (m *mockContent) HasData() bool {
	return len(m.data) > 0
}

func newMockContent(contentType MessageType, size int64, valid bool) *mockContent {
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i % 256)
	}

	return &mockContent{
		contentType: contentType,
		size:        size,
		data:        data,
		valid:       valid,
	}
}

func TestNewResourceManager(t *testing.T) {
	tests := []struct {
		name          string
		config        ResourceManagerConfig
		expectedImage int64
		expectedFile  int64
		expectedText  int64
	}{
		{
			name:          "default configuration",
			config:        ResourceManagerConfig{},
			expectedImage: 10 * 1024 * 1024,
			expectedFile:  50 * 1024 * 1024,
			expectedText:  1 * 1024 * 1024,
		},
		{
			name: "custom configuration",
			config: ResourceManagerConfig{
				MaxImageSize: 5 * 1024 * 1024,
				MaxFileSize:  25 * 1024 * 1024,
				MaxTextSize:  512 * 1024,
			},
			expectedImage: 5 * 1024 * 1024,
			expectedFile:  25 * 1024 * 1024,
			expectedText:  512 * 1024,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rm := NewResourceManager(tt.config)
			defer func() {
				if err := rm.Close(); err != nil {
					t.Logf("Failed to close resource manager: %v", err)
				}
			}()

			if rm.maxImageSize != tt.expectedImage {
				t.Errorf("Expected max image size %d, got %d", tt.expectedImage, rm.maxImageSize)
			}
			if rm.maxFileSize != tt.expectedFile {
				t.Errorf("Expected max file size %d, got %d", tt.expectedFile, rm.maxFileSize)
			}
			if rm.maxTextSize != tt.expectedText {
				t.Errorf("Expected max text size %d, got %d", tt.expectedText, rm.maxTextSize)
			}

			// Check that temp directory was created
			if _, err := os.Stat(rm.tempStoragePath); os.IsNotExist(err) {
				t.Errorf("Temp directory was not created: %s", rm.tempStoragePath)
			}
		})
	}
}

func TestResourceManager_ValidateContent(t *testing.T) {
	config := ResourceManagerConfig{
		MaxImageSize: 1024,
		MaxFileSize:  2048,
		MaxTextSize:  512,
	}
	rm := NewResourceManager(config)
	defer func() {
		if err := rm.Close(); err != nil {
			t.Logf("Failed to close resource manager: %v", err)
		}
	}()

	tests := []struct {
		name        string
		content     MessageContent
		expectError bool
	}{
		{
			name:        "valid text content",
			content:     newMockContent(MessageTypeText, 256, true),
			expectError: false,
		},
		{
			name:        "text content too large",
			content:     newMockContent(MessageTypeText, 1024, true),
			expectError: true,
		},
		{
			name:        "valid image content",
			content:     newMockContent(MessageTypeImage, 512, true),
			expectError: false,
		},
		{
			name:        "image content too large",
			content:     newMockContent(MessageTypeImage, 2048, true),
			expectError: true,
		},
		{
			name:        "valid file content",
			content:     newMockContent(MessageTypeFile, 1024, true),
			expectError: false,
		},
		{
			name:        "file content too large",
			content:     newMockContent(MessageTypeFile, 4096, true),
			expectError: true,
		},
		{
			name:        "invalid content",
			content:     newMockContent(MessageTypeText, 256, false),
			expectError: true,
		},
		{
			name:        "nil content",
			content:     nil,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := rm.ValidateContent(tt.content)
			if tt.expectError && err == nil {
				t.Error("Expected error, got nil")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error, got: %v", err)
			}
		})
	}
}

func TestResourceManager_ValidateMessage(t *testing.T) {
	config := ResourceManagerConfig{
		MaxImageSize: 1024,
		MaxFileSize:  2048,
		MaxTextSize:  512,
	}
	rm := NewResourceManager(config)
	defer func() {
		if err := rm.Close(); err != nil {
			t.Logf("Failed to close resource manager: %v", err)
		}
	}()

	tests := []struct {
		name        string
		message     Message
		expectError bool
	}{
		{
			name: "valid message",
			message: Message{
				Content: []MessageContent{
					newMockContent(MessageTypeText, 256, true),
					newMockContent(MessageTypeImage, 512, true),
				},
			},
			expectError: false,
		},
		{
			name: "message with oversized content",
			message: Message{
				Content: []MessageContent{
					newMockContent(MessageTypeText, 1024, true),
				},
			},
			expectError: true,
		},
		{
			name: "message with nil content",
			message: Message{
				Content: []MessageContent{
					newMockContent(MessageTypeText, 256, true),
					nil,
				},
			},
			expectError: true,
		},
		{
			name: "empty message",
			message: Message{
				Content: []MessageContent{},
			},
			expectError: true,
		},
		{
			name: "message exceeding total size limit",
			message: Message{
				Content: []MessageContent{
					newMockContent(MessageTypeText, 512, true),   // Max text
					newMockContent(MessageTypeImage, 1024, true), // Max image
					newMockContent(MessageTypeFile, 2048, true),  // Max file
					newMockContent(MessageTypeText, 1, true),     // Should exceed total
				},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := rm.ValidateMessage(tt.message)
			if tt.expectError && err == nil {
				t.Error("Expected error, got nil")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error, got: %v", err)
			}
		})
	}
}

func TestResourceManager_EstimateMemoryUsage(t *testing.T) {
	rm := NewResourceManager(ResourceManagerConfig{})
	defer func() {
		if err := rm.Close(); err != nil {
			t.Logf("Failed to close resource manager: %v", err)
		}
	}()

	tests := []struct {
		name            string
		message         Message
		expectedMinimum int64
	}{
		{
			name: "text message",
			message: Message{
				Content: []MessageContent{
					newMockContent(MessageTypeText, 100, true),
				},
			},
			expectedMinimum: 1200, // 100 + 100 overhead + 1000 base
		},
		{
			name: "image message",
			message: Message{
				Content: []MessageContent{
					newMockContent(MessageTypeImage, 1000, true),
				},
			},
			expectedMinimum: 4000, // 1000 * 2 + 1000 overhead + 1000 base
		},
		{
			name: "mixed content message",
			message: Message{
				Content: []MessageContent{
					newMockContent(MessageTypeText, 100, true),
					newMockContent(MessageTypeImage, 500, true),
				},
			},
			expectedMinimum: 2700, // (100+100) + (500*2+1000) + 1000
		},
		{
			name: "empty content",
			message: Message{
				Content: []MessageContent{},
			},
			expectedMinimum: 1000, // Base overhead only
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage := rm.EstimateMemoryUsage(tt.message)
			if usage < tt.expectedMinimum {
				t.Errorf("Expected memory usage >= %d, got %d", tt.expectedMinimum, usage)
			}
		})
	}
}

func TestResourceManager_TempFileOperations(t *testing.T) {
	// Create a custom temp directory for testing
	testTempDir := filepath.Join(os.TempDir(), fmt.Sprintf("test-rm-%d", time.Now().UnixNano()))
	config := ResourceManagerConfig{
		TempStoragePath: testTempDir,
	}
	rm := NewResourceManager(config)
	defer func() {
		if err := rm.Close(); err != nil {
			t.Logf("Failed to close resource manager: %v", err)
		}
	}()
	defer func() {
		if err := os.RemoveAll(testTempDir); err != nil {
			t.Logf("Failed to remove temp dir: %v", err)
		}
	}()

	testData := []byte("Hello, World! This is test binary data.")

	// Test storing binary content
	path, err := rm.StoreBinaryContent(testData)
	if err != nil {
		t.Fatalf("Failed to store binary content: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Errorf("Temp file was not created: %s", path)
	}

	// Test loading binary content
	loadedData, err := rm.LoadBinaryContent(path)
	if err != nil {
		t.Fatalf("Failed to load binary content: %v", err)
	}

	if !bytes.Equal(testData, loadedData) {
		t.Errorf("Loaded data does not match stored data")
	}

	// Test metrics
	metrics := rm.GetMetrics()
	if metrics.TempFilesCreated == 0 {
		t.Error("Expected temp files created metric to be > 0")
	}

	// Test loading non-existent file
	_, err = rm.LoadBinaryContent(filepath.Join(testTempDir, "nonexistent.tmp"))
	if err == nil {
		t.Error("Expected error when loading non-existent file")
	}

	// Test security: attempt to load file outside temp directory
	_, err = rm.LoadBinaryContent("/etc/passwd")
	if err == nil {
		t.Error("Expected error when accessing file outside temp directory")
	}
}

func TestResourceManager_CleanupTempFiles(t *testing.T) {
	// Create a custom temp directory for testing
	testTempDir := filepath.Join(os.TempDir(), fmt.Sprintf("test-cleanup-%d", time.Now().UnixNano()))
	config := ResourceManagerConfig{
		TempStoragePath: testTempDir,
		RetentionPeriod: 100 * time.Millisecond, // Very short retention for testing
	}
	rm := NewResourceManager(config)
	defer func() {
		if err := rm.Close(); err != nil {
			t.Logf("Failed to close resource manager: %v", err)
		}
	}()
	defer func() {
		if err := os.RemoveAll(testTempDir); err != nil {
			t.Logf("Failed to remove temp dir: %v", err)
		}
	}()

	// Create some test files
	testData := []byte("test data")
	path1, err := rm.StoreBinaryContent(testData)
	if err != nil {
		t.Fatalf("Failed to store binary content: %v", err)
	}

	path2, err := rm.StoreBinaryContent(testData)
	if err != nil {
		t.Fatalf("Failed to store binary content: %v", err)
	}

	// Verify files exist
	if _, err := os.Stat(path1); os.IsNotExist(err) {
		t.Error("First temp file should exist")
	}
	if _, err := os.Stat(path2); os.IsNotExist(err) {
		t.Error("Second temp file should exist")
	}

	// Wait for retention period to pass
	time.Sleep(150 * time.Millisecond)

	// Run cleanup
	err = rm.CleanupTempFiles()
	if err != nil {
		t.Fatalf("Failed to cleanup temp files: %v", err)
	}

	// Verify files were deleted
	if _, err := os.Stat(path1); !os.IsNotExist(err) {
		t.Error("First temp file should have been deleted")
	}
	if _, err := os.Stat(path2); !os.IsNotExist(err) {
		t.Error("Second temp file should have been deleted")
	}

	// Check metrics
	metrics := rm.GetMetrics()
	if metrics.TempFilesDeleted == 0 {
		t.Error("Expected temp files deleted metric to be > 0")
	}
}

func TestResourceManager_StreamContent(t *testing.T) {
	rm := NewResourceManager(ResourceManagerConfig{})
	defer func() {
		if err := rm.Close(); err != nil {
			t.Logf("Failed to close resource manager: %v", err)
		}
	}()

	tests := []struct {
		name        string
		content     MessageContent
		expectError bool
		expectedLen int
	}{
		{
			name:        "stream text content",
			content:     NewTextContent("Hello, World!"),
			expectError: false,
			expectedLen: 13,
		},
		{
			name:        "stream image content with data",
			content:     NewImageContentFromBytes([]byte("fake image data"), "image/png"),
			expectError: false,
			expectedLen: 15,
		},
		{
			name:        "stream file content with data",
			content:     NewFileContentFromBytes([]byte("file data"), "test.txt", "text/plain"),
			expectError: false,
			expectedLen: 9,
		},
		{
			name:        "stream nil content",
			content:     nil,
			expectError: true,
			expectedLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := rm.StreamContent(tt.content, &buf)

			if tt.expectError && err == nil {
				t.Error("Expected error, got nil")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error, got: %v", err)
			}

			if !tt.expectError && buf.Len() != tt.expectedLen {
				t.Errorf("Expected %d bytes streamed, got %d", tt.expectedLen, buf.Len())
			}
		})
	}

	// Test with nil writer
	content := NewTextContent("test")
	err := rm.StreamContent(content, nil)
	if err == nil {
		t.Error("Expected error when streaming to nil writer")
	}
}

func TestResourceManager_ReadContentStream(t *testing.T) {
	rm := NewResourceManager(ResourceManagerConfig{})
	defer func() {
		if err := rm.Close(); err != nil {
			t.Logf("Failed to close resource manager: %v", err)
		}
	}()

	tests := []struct {
		name        string
		data        string
		contentType MessageType
		expectError bool
	}{
		{
			name:        "read text stream",
			data:        "Hello, World!",
			contentType: MessageTypeText,
			expectError: false,
		},
		{
			name:        "read image stream",
			data:        "fake image data",
			contentType: MessageTypeImage,
			expectError: false,
		},
		{
			name:        "read file stream",
			data:        "file content",
			contentType: MessageTypeFile,
			expectError: false,
		},
		{
			name:        "unsupported content type",
			data:        "data",
			contentType: "unsupported",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.data)
			content, err := rm.ReadContentStream(reader, tt.contentType)

			if tt.expectError && err == nil {
				t.Error("Expected error, got nil")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error, got: %v", err)
			}

			if !tt.expectError && content == nil {
				t.Error("Expected content, got nil")
			}

			if !tt.expectError && content != nil {
				if content.Type() != tt.contentType {
					t.Errorf("Expected content type %s, got %s", tt.contentType, content.Type())
				}
			}
		})
	}

	// Test with nil reader
	_, err := rm.ReadContentStream(nil, MessageTypeText)
	if err == nil {
		t.Error("Expected error when reading from nil reader")
	}
}

func TestResourceManager_ShouldStream(t *testing.T) {
	config := ResourceManagerConfig{
		StreamThreshold: 1000,
	}
	rm := NewResourceManager(config)
	defer func() {
		if err := rm.Close(); err != nil {
			t.Logf("Failed to close resource manager: %v", err)
		}
	}()

	tests := []struct {
		name     string
		content  MessageContent
		expected bool
	}{
		{
			name:     "small content",
			content:  newMockContent(MessageTypeText, 500, true),
			expected: false,
		},
		{
			name:     "large content",
			content:  newMockContent(MessageTypeImage, 2000, true),
			expected: true,
		},
		{
			name:     "threshold content",
			content:  newMockContent(MessageTypeFile, 1000, true),
			expected: true,
		},
		{
			name:     "nil content",
			content:  nil,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := rm.ShouldStream(tt.content)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestResourceManager_UpdateConfig(t *testing.T) {
	rm := NewResourceManager(ResourceManagerConfig{})
	defer func() {
		if err := rm.Close(); err != nil {
			t.Logf("Failed to close resource manager: %v", err)
		}
	}()

	newConfig := ResourceManagerConfig{
		MaxImageSize:    5 * 1024 * 1024,
		MaxFileSize:     25 * 1024 * 1024,
		MaxTextSize:     512 * 1024,
		StreamThreshold: 2 * 1024 * 1024,
	}

	err := rm.UpdateConfig(newConfig)
	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	if rm.maxImageSize != newConfig.MaxImageSize {
		t.Errorf("Expected max image size %d, got %d", newConfig.MaxImageSize, rm.maxImageSize)
	}
	if rm.maxFileSize != newConfig.MaxFileSize {
		t.Errorf("Expected max file size %d, got %d", newConfig.MaxFileSize, rm.maxFileSize)
	}
	if rm.maxTextSize != newConfig.MaxTextSize {
		t.Errorf("Expected max text size %d, got %d", newConfig.MaxTextSize, rm.maxTextSize)
	}
	if rm.streamThreshold != newConfig.StreamThreshold {
		t.Errorf("Expected stream threshold %d, got %d", newConfig.StreamThreshold, rm.streamThreshold)
	}
}

func TestResourceManager_ValidateResourceHealth(t *testing.T) {
	// Test with valid temp directory
	testTempDir := filepath.Join(os.TempDir(), fmt.Sprintf("test-health-%d", time.Now().UnixNano()))
	config := ResourceManagerConfig{
		TempStoragePath: testTempDir,
	}
	rm := NewResourceManager(config)
	defer func() {
		if err := rm.Close(); err != nil {
			t.Logf("Failed to close resource manager: %v", err)
		}
	}()
	defer func() {
		if err := os.RemoveAll(testTempDir); err != nil {
			t.Logf("Failed to remove temp dir: %v", err)
		}
	}()

	err := rm.ValidateResourceHealth()
	if err != nil {
		t.Errorf("Expected no error for valid health check, got: %v", err)
	}

	// Test with invalid temp directory (read-only parent)
	// Note: This test might not work on all systems due to permission requirements
	invalidTempDir := "/proc/invalid-temp-dir"
	rmInvalid := &ResourceManager{
		tempStoragePath: invalidTempDir,
	}

	err = rmInvalid.ValidateResourceHealth()
	if err == nil {
		t.Log("Warning: Expected error for invalid temp directory, but got none (this might be system-dependent)")
	}
}

func TestResourceManager_GetMetrics(t *testing.T) {
	config := ResourceManagerConfig{
		MaxImageSize:    1024,
		MaxFileSize:     2048,
		MaxTextSize:     512,
		StreamThreshold: 1500,
	}
	rm := NewResourceManager(config)
	defer func() {
		if err := rm.Close(); err != nil {
			t.Logf("Failed to close resource manager: %v", err)
		}
	}()

	// Create some temp files to affect metrics
	testData := []byte("test data for metrics")
	_, err := rm.StoreBinaryContent(testData)
	if err != nil {
		t.Fatalf("Failed to store binary content: %v", err)
	}

	metrics := rm.GetMetrics()

	if metrics.MaxImageSize != config.MaxImageSize {
		t.Errorf("Expected max image size %d, got %d", config.MaxImageSize, metrics.MaxImageSize)
	}
	if metrics.MaxFileSize != config.MaxFileSize {
		t.Errorf("Expected max file size %d, got %d", config.MaxFileSize, metrics.MaxFileSize)
	}
	if metrics.MaxTextSize != config.MaxTextSize {
		t.Errorf("Expected max text size %d, got %d", config.MaxTextSize, metrics.MaxTextSize)
	}
	if metrics.StreamThreshold != config.StreamThreshold {
		t.Errorf("Expected stream threshold %d, got %d", config.StreamThreshold, metrics.StreamThreshold)
	}
	if metrics.TempFilesCreated == 0 {
		t.Error("Expected temp files created > 0")
	}
}

func TestResourceManager_ConcurrentOperations(t *testing.T) {
	testTempDir := filepath.Join(os.TempDir(), fmt.Sprintf("test-concurrent-%d", time.Now().UnixNano()))
	config := ResourceManagerConfig{
		TempStoragePath: testTempDir,
	}
	rm := NewResourceManager(config)
	defer func() {
		if err := rm.Close(); err != nil {
			t.Logf("Failed to close resource manager: %v", err)
		}
	}()
	defer func() {
		if err := os.RemoveAll(testTempDir); err != nil {
			t.Logf("Failed to remove temp dir: %v", err)
		}
	}()

	const numGoroutines = 10
	const numOperations = 5

	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines*numOperations)

	// Test concurrent temp file operations
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			for j := 0; j < numOperations; j++ {
				testData := []byte(fmt.Sprintf("test data %d-%d", id, j))

				// Store file
				path, err := rm.StoreBinaryContent(testData)
				if err != nil {
					errors <- fmt.Errorf("goroutine %d: failed to store: %v", id, err)
					continue
				}

				// Load file
				loadedData, err := rm.LoadBinaryContent(path)
				if err != nil {
					errors <- fmt.Errorf("goroutine %d: failed to load: %v", id, err)
					continue
				}

				if !bytes.Equal(testData, loadedData) {
					errors <- fmt.Errorf("goroutine %d: data mismatch", id)
				}
			}
		}(i)
	}

	wg.Wait()
	close(errors)

	// Check for any errors
	for err := range errors {
		t.Errorf("Concurrent operation error: %v", err)
	}

	// Verify metrics are consistent
	metrics := rm.GetMetrics()
	if metrics.TempFilesCreated != int64(numGoroutines*numOperations) {
		t.Errorf("Expected %d temp files created, got %d", numGoroutines*numOperations, metrics.TempFilesCreated)
	}
}

func TestResourceManager_EdgeCases(t *testing.T) {
	rm := NewResourceManager(ResourceManagerConfig{})
	defer func() {
		if err := rm.Close(); err != nil {
			t.Logf("Failed to close resource manager: %v", err)
		}
	}()

	t.Run("zero-size content", func(t *testing.T) {
		zeroContent := newMockContent(MessageTypeText, 0, true)
		err := rm.ValidateContent(zeroContent)
		if err != nil {
			t.Errorf("Expected no error for zero-size content, got: %v", err)
		}
	})

	t.Run("empty binary data storage", func(t *testing.T) {
		path, err := rm.StoreBinaryContent([]byte{})
		if err != nil {
			t.Errorf("Expected no error for empty data storage, got: %v", err)
		}

		data, err := rm.LoadBinaryContent(path)
		if err != nil {
			t.Errorf("Expected no error loading empty data, got: %v", err)
		}

		if len(data) != 0 {
			t.Errorf("Expected empty data, got %d bytes", len(data))
		}
	})

	t.Run("stream empty content", func(t *testing.T) {
		var buf bytes.Buffer
		err := rm.StreamContent(NewTextContent(""), &buf)
		// This might error due to text content validation, which is expected
		if err == nil && buf.Len() != 0 {
			t.Errorf("Expected empty buffer for empty content stream")
		}
	})
}

func TestResourceManager_Close(t *testing.T) {
	testTempDir := filepath.Join(os.TempDir(), fmt.Sprintf("test-close-%d", time.Now().UnixNano()))
	config := ResourceManagerConfig{
		TempStoragePath: testTempDir,
		CleanupInterval: 10 * time.Millisecond,
	}
	rm := NewResourceManager(config)

	// Create a temp file
	testData := []byte("test data for close")
	path, err := rm.StoreBinaryContent(testData)
	if err != nil {
		t.Fatalf("Failed to store binary content: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Error("Temp file should exist before close")
	}

	// Close the resource manager
	err = rm.Close()
	if err != nil {
		t.Errorf("Expected no error on close, got: %v", err)
	}

	// Cleanup temp directory
	if err := os.RemoveAll(testTempDir); err != nil {
		t.Logf("Failed to remove temp dir: %v", err)
	}
}

// Benchmark tests
func BenchmarkResourceManager_ValidateContent(b *testing.B) {
	rm := NewResourceManager(ResourceManagerConfig{})
	defer func() {
		if err := rm.Close(); err != nil {
			b.Logf("Failed to close resource manager: %v", err)
		}
	}()

	content := newMockContent(MessageTypeText, 1024, true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := rm.ValidateContent(content); err != nil {
			b.Fatalf("ValidateContent failed: %v", err)
		}
	}
}

func BenchmarkResourceManager_EstimateMemoryUsage(b *testing.B) {
	rm := NewResourceManager(ResourceManagerConfig{})
	defer func() {
		if err := rm.Close(); err != nil {
			b.Logf("Failed to close resource manager: %v", err)
		}
	}()

	message := Message{
		Content: []MessageContent{
			newMockContent(MessageTypeText, 1024, true),
			newMockContent(MessageTypeImage, 2048, true),
			newMockContent(MessageTypeFile, 1536, true),
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rm.EstimateMemoryUsage(message)
	}
}

func BenchmarkResourceManager_TempFileOperations(b *testing.B) {
	testTempDir := filepath.Join(os.TempDir(), fmt.Sprintf("bench-temp-%d", time.Now().UnixNano()))
	config := ResourceManagerConfig{
		TempStoragePath: testTempDir,
	}
	rm := NewResourceManager(config)
	defer func() {
		if err := rm.Close(); err != nil {
			b.Logf("Failed to close resource manager: %v", err)
		}
	}()
	defer func() {
		if err := os.RemoveAll(testTempDir); err != nil {
			b.Logf("Failed to remove temp dir: %v", err)
		}
	}()

	testData := make([]byte, 1024) // 1KB test data
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		path, err := rm.StoreBinaryContent(testData)
		if err != nil {
			b.Errorf("Failed to store: %v", err)
		}

		_, err = rm.LoadBinaryContent(path)
		if err != nil {
			b.Errorf("Failed to load: %v", err)
		}
	}
}

func BenchmarkResourceManager_StreamContent(b *testing.B) {
	rm := NewResourceManager(ResourceManagerConfig{})
	defer func() {
		if err := rm.Close(); err != nil {
			b.Logf("Failed to close resource manager: %v", err)
		}
	}()

	content := NewTextContent(strings.Repeat("Hello World! ", 100))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		err := rm.StreamContent(content, &buf)
		if err != nil {
			b.Errorf("Failed to stream: %v", err)
		}
	}
}
