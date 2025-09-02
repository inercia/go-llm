package llm

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ResourceManager provides comprehensive resource management for multi-modal content
// including size limits, validation, temporary file management, and memory-efficient streaming
type ResourceManager struct {
	// Configuration
	maxImageSize    int64         // Maximum size for image content in bytes
	maxFileSize     int64         // Maximum size for file content in bytes
	maxTextSize     int64         // Maximum size for text content in bytes
	tempStoragePath string        // Directory for temporary file storage
	cleanupInterval time.Duration // Interval for automatic cleanup
	retentionPeriod time.Duration // How long to keep temp files
	streamThreshold int64         // Size threshold for streaming operations

	// State management
	mu              sync.RWMutex
	tempFileCounter int64
	cleanupTicker   *time.Ticker
	cleanupStop     chan bool

	// Metrics
	tempFilesCreated int64
	tempFilesDeleted int64
}

// ResourceManagerConfig holds configuration for the ResourceManager
type ResourceManagerConfig struct {
	MaxImageSize    int64         `json:"max_image_size"`
	MaxFileSize     int64         `json:"max_file_size"`
	MaxTextSize     int64         `json:"max_text_size"`
	TempStoragePath string        `json:"temp_storage_path"`
	CleanupInterval time.Duration `json:"cleanup_interval"`
	RetentionPeriod time.Duration `json:"retention_period"`
	StreamThreshold int64         `json:"stream_threshold"`
}

// NewResourceManager creates a new ResourceManager instance with the given configuration
func NewResourceManager(config ResourceManagerConfig) *ResourceManager {
	// Set default values if not provided
	if config.MaxImageSize <= 0 {
		config.MaxImageSize = 10 * 1024 * 1024 // 10MB default
	}
	if config.MaxFileSize <= 0 {
		config.MaxFileSize = 50 * 1024 * 1024 // 50MB default
	}
	if config.MaxTextSize <= 0 {
		config.MaxTextSize = 1 * 1024 * 1024 // 1MB default
	}
	if config.TempStoragePath == "" {
		config.TempStoragePath = filepath.Join(os.TempDir(), "llm-agent-temp")
	}
	if config.CleanupInterval <= 0 {
		config.CleanupInterval = 1 * time.Hour // 1 hour default
	}
	if config.RetentionPeriod <= 0 {
		config.RetentionPeriod = 24 * time.Hour // 24 hours default
	}
	if config.StreamThreshold <= 0 {
		config.StreamThreshold = 5 * 1024 * 1024 // 5MB default
	}

	rm := &ResourceManager{
		maxImageSize:    config.MaxImageSize,
		maxFileSize:     config.MaxFileSize,
		maxTextSize:     config.MaxTextSize,
		tempStoragePath: config.TempStoragePath,
		cleanupInterval: config.CleanupInterval,
		retentionPeriod: config.RetentionPeriod,
		streamThreshold: config.StreamThreshold,
		cleanupStop:     make(chan bool, 1),
	}

	// Create temp directory if it doesn't exist
	if err := os.MkdirAll(rm.tempStoragePath, 0750); err != nil {
		// Note: In production, you might want to handle this error differently
		fmt.Printf("Warning: failed to create temp directory %s: %v\n", rm.tempStoragePath, err)
	}

	// Start automatic cleanup
	rm.startCleanup()

	return rm
}

// Close stops the resource manager and cleanup any resources
func (rm *ResourceManager) Close() error {
	if rm.cleanupTicker != nil {
		rm.cleanupTicker.Stop()
		close(rm.cleanupStop)
	}
	return rm.CleanupTempFiles()
}

// ValidateContent checks if content meets size limits and other constraints
func (rm *ResourceManager) ValidateContent(content MessageContent) error {
	if content == nil {
		return fmt.Errorf("content cannot be nil")
	}

	// Basic validation
	if err := content.Validate(); err != nil {
		return fmt.Errorf("content validation failed: %w", err)
	}

	// Size validation based on content type
	size := content.Size()
	contentType := content.Type()

	switch contentType {
	case MessageTypeText:
		if size > rm.maxTextSize {
			return fmt.Errorf("text content size %d bytes exceeds maximum %d bytes", size, rm.maxTextSize)
		}
	case MessageTypeImage:
		if size > rm.maxImageSize {
			return fmt.Errorf("image content size %d bytes exceeds maximum %d bytes", size, rm.maxImageSize)
		}
	case MessageTypeFile:
		if size > rm.maxFileSize {
			return fmt.Errorf("file content size %d bytes exceeds maximum %d bytes", size, rm.maxFileSize)
		}
	default:
		return fmt.Errorf("unsupported content type: %s", contentType)
	}

	return nil
}

// ValidateMessage validates an entire message including all its content
func (rm *ResourceManager) ValidateMessage(message Message) error {
	if len(message.Content) == 0 {
		return fmt.Errorf("message has no content")
	}

	totalSize := int64(0)
	for i, content := range message.Content {
		if content == nil {
			return fmt.Errorf("content item %d is nil", i)
		}

		if err := rm.ValidateContent(content); err != nil {
			return fmt.Errorf("content item %d validation failed: %w", i, err)
		}

		totalSize += content.Size()
	}

	// Check total message size (sum of all limits)
	maxTotalSize := rm.maxTextSize + rm.maxImageSize + rm.maxFileSize
	if totalSize > maxTotalSize {
		return fmt.Errorf("message total size %d bytes exceeds maximum %d bytes", totalSize, maxTotalSize)
	}

	return nil
}

// EstimateMemoryUsage estimates the memory usage of processing a message
func (rm *ResourceManager) EstimateMemoryUsage(message Message) int64 {
	totalSize := int64(0)

	for _, content := range message.Content {
		if content == nil {
			continue
		}

		contentSize := content.Size()

		// Estimate additional overhead based on content type
		switch content.Type() {
		case MessageTypeText:
			// Text has minimal overhead (string storage)
			totalSize += contentSize + 100
		case MessageTypeImage:
			// Images may need decoding buffers
			totalSize += contentSize*2 + 1000
		case MessageTypeFile:
			// Files may need processing buffers
			totalSize += contentSize + 500
		}
	}

	// Add base message overhead
	totalSize += 1000

	return totalSize
}

// StoreBinaryContent stores binary data to a temporary file and returns the file path
func (rm *ResourceManager) StoreBinaryContent(data []byte) (string, error) {
	rm.mu.Lock()
	rm.tempFileCounter++
	counter := rm.tempFileCounter
	rm.mu.Unlock()

	// Create unique filename with timestamp and counter
	timestamp := time.Now().UnixNano()
	filename := fmt.Sprintf("content_%d_%d.tmp", timestamp, counter)
	filepath := filepath.Join(rm.tempStoragePath, filename)

	// Write data to file
	file, err := rm.secureCreateFile(filepath)
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer func() { _ = file.Close() }()

	if _, err := file.Write(data); err != nil {
		_ = os.Remove(filepath) // Clean up on failure
		return "", fmt.Errorf("failed to write to temp file: %w", err)
	}

	// Update metrics
	rm.mu.Lock()
	rm.tempFilesCreated++
	rm.mu.Unlock()

	return filepath, nil
}

// LoadBinaryContent loads binary data from a temporary file
func (rm *ResourceManager) LoadBinaryContent(path string) ([]byte, error) {
	// Validate path is within temp storage directory for security
	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("invalid file path: %w", err)
	}

	absTempPath, err := filepath.Abs(rm.tempStoragePath)
	if err != nil {
		return nil, fmt.Errorf("invalid temp storage path: %w", err)
	}

	rel, err := filepath.Rel(absTempPath, absPath)
	if err != nil || filepath.IsAbs(rel) || len(rel) > 0 && rel[0] == '.' {
		return nil, fmt.Errorf("file path is outside temp storage directory")
	}

	// Read file
	data, err := rm.secureReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read temp file: %w", err)
	}

	return data, nil
}

// CleanupTempFiles removes old temporary files based on retention policy
func (rm *ResourceManager) CleanupTempFiles() error {
	entries, err := os.ReadDir(rm.tempStoragePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // Directory doesn't exist, nothing to clean
		}
		return fmt.Errorf("failed to read temp directory: %w", err)
	}

	cutoff := time.Now().Add(-rm.retentionPeriod)
	filesDeleted := int64(0)

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			continue // Skip files we can't get info for
		}

		// Delete files older than retention period
		if info.ModTime().Before(cutoff) {
			path := filepath.Join(rm.tempStoragePath, entry.Name())
			if err := os.Remove(path); err == nil {
				filesDeleted++
			}
		}
	}

	// Update metrics
	rm.mu.Lock()
	rm.tempFilesDeleted += filesDeleted
	rm.mu.Unlock()

	return nil
}

// StreamContent writes content to the provided writer in a memory-efficient way
func (rm *ResourceManager) StreamContent(content MessageContent, writer io.Writer) error {
	if content == nil {
		return fmt.Errorf("content cannot be nil")
	}

	if writer == nil {
		return fmt.Errorf("writer cannot be nil")
	}

	switch c := content.(type) {
	case *TextContent:
		_, err := writer.Write([]byte(c.Text))
		return err

	case *ImageContent:
		if c.HasData() {
			_, err := writer.Write(c.Data)
			return err
		}
		// For URL-based images, we would need to fetch and stream
		return fmt.Errorf("streaming URL-based image content not implemented")

	case *FileContent:
		if c.HasData() {
			_, err := writer.Write(c.Data)
			return err
		}
		// For URL-based files, we would need to fetch and stream
		return fmt.Errorf("streaming URL-based file content not implemented")

	default:
		return fmt.Errorf("unsupported content type for streaming: %T", content)
	}
}

// ReadContentStream reads content from a reader and creates appropriate MessageContent
func (rm *ResourceManager) ReadContentStream(reader io.Reader, contentType MessageType) (MessageContent, error) {
	if reader == nil {
		return nil, fmt.Errorf("reader cannot be nil")
	}

	// Read data from stream
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read from stream: %w", err)
	}

	// Create appropriate content type
	switch contentType {
	case MessageTypeText:
		return NewTextContent(string(data)), nil

	case MessageTypeImage:
		// For images, we'd need to detect MIME type or have it provided
		return NewImageContentFromBytes(data, "application/octet-stream"), nil

	case MessageTypeFile:
		// For files, we'd need filename and MIME type provided
		return NewFileContentFromBytes(data, "stream.dat", "application/octet-stream"), nil

	default:
		return nil, fmt.Errorf("unsupported content type: %s", contentType)
	}
}

// GetMetrics returns current resource management metrics
func (rm *ResourceManager) GetMetrics() ResourceMetrics {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	return ResourceMetrics{
		TempFilesCreated: rm.tempFilesCreated,
		TempFilesDeleted: rm.tempFilesDeleted,
		MaxImageSize:     rm.maxImageSize,
		MaxFileSize:      rm.maxFileSize,
		MaxTextSize:      rm.maxTextSize,
		StreamThreshold:  rm.streamThreshold,
	}
}

// ResourceMetrics holds metrics about resource usage
type ResourceMetrics struct {
	TempFilesCreated int64 `json:"temp_files_created"`
	TempFilesDeleted int64 `json:"temp_files_deleted"`
	MaxImageSize     int64 `json:"max_image_size"`
	MaxFileSize      int64 `json:"max_file_size"`
	MaxTextSize      int64 `json:"max_text_size"`
	StreamThreshold  int64 `json:"stream_threshold"`
}

// ShouldStream returns whether content should be streamed based on size threshold
func (rm *ResourceManager) ShouldStream(content MessageContent) bool {
	if content == nil {
		return false
	}
	return content.Size() >= rm.streamThreshold
}

// GetTempFilePath returns the full path for a temp file with the given name
func (rm *ResourceManager) GetTempFilePath(filename string) string {
	return filepath.Join(rm.tempStoragePath, filename)
}

// GetAvailableDiskSpace returns available disk space in the temp directory
func (rm *ResourceManager) GetAvailableDiskSpace() (int64, error) {
	// This is a simplified implementation
	// In production, you'd want to use syscalls to get actual disk space
	stat, err := os.Stat(rm.tempStoragePath)
	if err != nil {
		return 0, fmt.Errorf("failed to stat temp directory: %w", err)
	}

	if !stat.IsDir() {
		return 0, fmt.Errorf("temp storage path is not a directory")
	}

	// Return a placeholder value (in production, use proper disk space detection)
	return 1024 * 1024 * 1024, nil // 1GB placeholder
}

// startCleanup starts the automatic cleanup process
func (rm *ResourceManager) startCleanup() {
	rm.cleanupTicker = time.NewTicker(rm.cleanupInterval)

	go func() {
		for {
			select {
			case <-rm.cleanupTicker.C:
				_ = rm.CleanupTempFiles()
			case <-rm.cleanupStop:
				return
			}
		}
	}()
}

// UpdateConfig updates the resource manager configuration
func (rm *ResourceManager) UpdateConfig(config ResourceManagerConfig) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	if config.MaxImageSize > 0 {
		rm.maxImageSize = config.MaxImageSize
	}
	if config.MaxFileSize > 0 {
		rm.maxFileSize = config.MaxFileSize
	}
	if config.MaxTextSize > 0 {
		rm.maxTextSize = config.MaxTextSize
	}
	if config.StreamThreshold > 0 {
		rm.streamThreshold = config.StreamThreshold
	}

	return nil
}

// ValidateResourceHealth performs a health check of the resource management system
func (rm *ResourceManager) ValidateResourceHealth() error {
	// Check temp directory exists and is writable
	if err := os.MkdirAll(rm.tempStoragePath, 0750); err != nil {
		return fmt.Errorf("temp directory not accessible: %w", err)
	}

	// Try creating a test file
	testFile := filepath.Join(rm.tempStoragePath, "health_check.tmp")
	file, err := rm.secureCreateFile(testFile)
	if err != nil {
		return fmt.Errorf("temp directory not writable: %w", err)
	}
	_ = file.Close()
	_ = os.Remove(testFile) // Clean up test file

	// Check available disk space
	availableSpace, err := rm.GetAvailableDiskSpace()
	if err != nil {
		return fmt.Errorf("failed to check disk space: %w", err)
	}

	// Warn if disk space is low (less than 100MB)
	if availableSpace < 100*1024*1024 {
		return fmt.Errorf("low disk space: only %d bytes available", availableSpace)
	}

	return nil
}

// validateFilePath ensures the file path is safe and within allowed directories
func (rm *ResourceManager) validateFilePath(path string) error {
	// Clean the path to resolve any .. or . components
	cleanPath := filepath.Clean(path)

	// Check for path traversal attempts
	if strings.Contains(cleanPath, "..") {
		return fmt.Errorf("path traversal detected: %s", path)
	}

	// Ensure path is within temp storage directory
	absPath, err := filepath.Abs(cleanPath)
	if err != nil {
		return fmt.Errorf("invalid file path: %w", err)
	}

	absTempPath, err := filepath.Abs(rm.tempStoragePath)
	if err != nil {
		return fmt.Errorf("invalid temp storage path: %w", err)
	}

	// Check if the path is within the temp directory
	rel, err := filepath.Rel(absTempPath, absPath)
	if err != nil || filepath.IsAbs(rel) || strings.HasPrefix(rel, "..") {
		return fmt.Errorf("file path is outside temp storage directory: %s", path)
	}

	return nil
}

// secureCreateFile creates a file with path validation
func (rm *ResourceManager) secureCreateFile(filepath string) (*os.File, error) {
	if err := rm.validateFilePath(filepath); err != nil {
		return nil, err
	}
	// #nosec G304 - Path is validated above to prevent traversal attacks
	return os.Create(filepath)
}

// secureReadFile reads a file with path validation
func (rm *ResourceManager) secureReadFile(path string) ([]byte, error) {
	if err := rm.validateFilePath(path); err != nil {
		return nil, err
	}
	// #nosec G304 - Path is validated above to prevent traversal attacks
	return os.ReadFile(path)
}
