package llm

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"
)

// Test data for security testing
var (
	// Safe test image (1x1 PNG)
	testImagePNG = []byte{
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
		0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
		0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
		0x54, 0x08, 0xD7, 0x63, 0xF8, 0x0F, 0x00, 0x00,
		0x01, 0x00, 0x01, 0x5C, 0xC2, 0x88, 0x05, 0x00,
		0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
		0x42, 0x60, 0x82,
	}

	// Safe test JPEG header
	testImageJPEG = []byte{
		0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
		0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00, 0x48,
	}

	// Malicious SVG with script
	maliciousSVG = []byte(`<svg xmlns="http://www.w3.org/2000/svg">
		<script>alert('xss')</script>
		<circle cx="50" cy="50" r="40"/>
	</svg>`)

	// EICAR test virus signature
	eicarTestVirus = []byte("X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*")

	// Test PDF header
	testPDFHeader = []byte("%PDF-1.4")

	// Malicious PDF with JavaScript
	maliciousPDF = []byte("%PDF-1.4\n/JavaScript action")

	// Test JSON data
	testJSONValid       = []byte(`{"key": "value", "number": 123}`)
	testJSONMalicious   = []byte(`{"__proto__": {"isAdmin": true}}`)
	testJSONDeepNesting = generateDeeplyNestedJSON(50)

	// CSV injection attempts
	testCSVSafe      = []byte("name,value\ntest,123")
	testCSVMalicious = []byte("name,value\n=cmd|'/C calc'!A1,123")
)

func TestDefaultSecurityConfig(t *testing.T) {
	config := DefaultSecurityConfig()

	// Test that all required fields are set
	if config.MaxImageSize == 0 {
		t.Error("MaxImageSize should be set")
	}
	if config.MaxFileSize == 0 {
		t.Error("MaxFileSize should be set")
	}
	if config.MaxTotalSize == 0 {
		t.Error("MaxTotalSize should be set")
	}
	if len(config.AllowedImageMIMEs) == 0 {
		t.Error("AllowedImageMIMEs should not be empty")
	}
	if len(config.AllowedFileMIMEs) == 0 {
		t.Error("AllowedFileMIMEs should not be empty")
	}

	// Test default security policies
	if !config.EnableMalwareScanning {
		t.Error("Malware scanning should be enabled by default")
	}
	if !config.EnablePathValidation {
		t.Error("Path validation should be enabled by default")
	}
	if !config.EnableContentScan {
		t.Error("Content scanning should be enabled by default")
	}
}

func TestSecurityValidator_ValidateImageContent(t *testing.T) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	tests := []struct {
		name        string
		image       *ImageContent
		expectError bool
		errorType   string
	}{
		{
			name: "valid PNG image",
			image: &ImageContent{
				Data:     testImagePNG,
				MimeType: "image/png",
			},
			expectError: false,
		},
		{
			name: "valid JPEG image",
			image: &ImageContent{
				Data:     testImageJPEG,
				MimeType: "image/jpeg",
			},
			expectError: false,
		},
		{
			name: "invalid MIME type",
			image: &ImageContent{
				Data:     testImagePNG,
				MimeType: "application/evil",
			},
			expectError: true,
			errorType:   "MIME_REJECTED",
		},
		{
			name: "MIME type mismatch",
			image: &ImageContent{
				Data:     testImagePNG,
				MimeType: "image/jpeg",
			},
			expectError: true,
			errorType:   "SIGNATURE_MISMATCH",
		},
		{
			name: "malicious SVG with script",
			image: &ImageContent{
				Data:     maliciousSVG,
				MimeType: "image/svg+xml",
			},
			expectError: true,
			errorType:   "MALICIOUS_CONTENT",
		},
		{
			name: "safe URL",
			image: &ImageContent{
				URL:      "https://example.com/image.png",
				MimeType: "image/png",
			},
			expectError: false,
		},
		{
			name: "dangerous URL scheme",
			image: &ImageContent{
				URL:      "javascript:alert('xss')",
				MimeType: "image/png",
			},
			expectError: true,
			errorType:   "URL_REJECTED",
		},
		{
			name: "private IP URL",
			image: &ImageContent{
				URL:      "http://192.168.1.1/image.png",
				MimeType: "image/png",
			},
			expectError: true,
			errorType:   "URL_REJECTED",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.ValidateContentSecurity(tt.image)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error for %s, but got none", tt.name)
				}
				// Check that the appropriate security event was logged
				if tt.errorType != "" {
					events := validator.GetAuditEventsByType(tt.errorType)
					if len(events) == 0 {
						t.Errorf("Expected security event of type %s to be logged", tt.errorType)
					}
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error for %s: %v", tt.name, err)
				}
			}
		})
	}
}

func TestSecurityValidator_ValidateFileContent(t *testing.T) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	tests := []struct {
		name        string
		file        *FileContent
		expectError bool
		errorType   string
	}{
		{
			name: "valid text file",
			file: &FileContent{
				Data:     []byte("Hello, World!"),
				MimeType: "text/plain",
				Filename: "test.txt",
				FileSize: 13, // Length of "Hello, World!"
			},
			expectError: false,
		},
		{
			name: "valid JSON file",
			file: &FileContent{
				Data:     testJSONValid,
				MimeType: "application/json",
				Filename: "data.json",
				FileSize: int64(len(testJSONValid)),
			},
			expectError: false,
		},
		{
			name: "invalid MIME type",
			file: &FileContent{
				Data:     []byte("test"),
				MimeType: "application/dangerous",
				Filename: "test.dangerous",
				FileSize: 4,
			},
			expectError: true,
			errorType:   "MIME_REJECTED",
		},
		{
			name: "path traversal filename",
			file: &FileContent{
				Data:     []byte("test"),
				MimeType: "text/plain",
				Filename: "../../../etc/passwd",
				FileSize: 4,
			},
			expectError: true,
			errorType:   "PATH_TRAVERSAL",
		},
		{
			name: "extension mismatch",
			file: &FileContent{
				Data:     testJSONValid,
				MimeType: "application/json",
				Filename: "data.txt", // Wrong extension
				FileSize: int64(len(testJSONValid)),
			},
			expectError: true,
			errorType:   "EXTENSION_MISMATCH",
		},
		{
			name: "EICAR virus signature",
			file: &FileContent{
				Data:     eicarTestVirus,
				MimeType: "text/plain",
				Filename: "virus.txt",
				FileSize: int64(len(eicarTestVirus)),
			},
			expectError: true,
			errorType:   "MALWARE_DETECTED",
		},
		{
			name: "malicious PDF",
			file: &FileContent{
				Data:     maliciousPDF,
				MimeType: "application/pdf",
				Filename: "malicious.pdf",
				FileSize: int64(len(maliciousPDF)),
			},
			expectError: true,
			errorType:   "MALWARE_DETECTED", // Files use MALWARE_DETECTED, images use MALICIOUS_CONTENT
		},
		{
			name: "JSON with prototype pollution",
			file: &FileContent{
				Data:     testJSONMalicious,
				MimeType: "application/json",
				Filename: "malicious.json",
				FileSize: int64(len(testJSONMalicious)),
			},
			expectError: true,
			errorType:   "SIGNATURE_MISMATCH",
		},
		{
			name: "CSV injection",
			file: &FileContent{
				Data:     testCSVMalicious,
				MimeType: "text/csv",
				Filename: "malicious.csv",
				FileSize: int64(len(testCSVMalicious)),
			},
			expectError: true,
			errorType:   "SIGNATURE_MISMATCH",
		},
		{
			name: "executable file URL",
			file: &FileContent{
				URL:      "https://example.com/malware.exe",
				MimeType: "application/pdf",
				Filename: "document.pdf",
				FileSize: 0, // URL-based, no data
			},
			expectError: true,
			errorType:   "URL_REJECTED",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.ValidateContentSecurity(tt.file)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error for %s, but got none", tt.name)
				}
				// Check that the appropriate security event was logged
				if tt.errorType != "" {
					events := validator.GetAuditEventsByType(tt.errorType)
					if len(events) == 0 {
						t.Errorf("Expected security event of type %s to be logged", tt.errorType)
					}
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error for %s: %v", tt.name, err)
				}
			}
		})
	}
}

func TestSecurityValidator_ValidateTextContent(t *testing.T) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	tests := []struct {
		name        string
		text        *TextContent
		expectError bool
		errorType   string
	}{
		{
			name: "safe text",
			text: &TextContent{
				Text: "Hello, this is safe text content.",
			},
			expectError: false,
		},
		{
			name: "text with script injection",
			text: &TextContent{
				Text: "Hello <script>alert('xss')</script>",
			},
			expectError: true,
			errorType:   "SUSPICIOUS_TEXT",
		},
		{
			name: "text with javascript scheme",
			text: &TextContent{
				Text: "Click here: javascript:alert('xss')",
			},
			expectError: true,
			errorType:   "SUSPICIOUS_TEXT",
		},
		{
			name: "text with SQL injection attempt",
			text: &TextContent{
				Text: "'; DROP TABLE users; --",
			},
			expectError: false, // SQL patterns are logged but not blocked for text
		},
		{
			name: "very large text",
			text: &TextContent{
				Text: strings.Repeat("A", int(101*1024*1024)), // 101MB
			},
			expectError: true,
			errorType:   "SIZE_EXCEEDED",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.ValidateContentSecurity(tt.text)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error for %s, but got none", tt.name)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error for %s: %v", tt.name, err)
				}
			}
		})
	}
}

func TestSecurityValidator_FilenameValidation(t *testing.T) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	tests := []struct {
		name        string
		filename    string
		expectError bool
	}{
		{"valid filename", "document.pdf", false},
		{"filename with spaces", "my document.pdf", false},
		{"path traversal", "../../../etc/passwd", true},
		{"relative path traversal", "..\\..\\windows\\system32", true},
		{"absolute path", "/etc/passwd", true},
		{"windows absolute path", "C:\\Windows\\System32\\calc.exe", true},
		{"null byte injection", "test\x00.pdf", true},
		{"newline injection", "test\n.pdf", true},
		{"tab injection", "test\t.pdf", true},
		{"carriage return injection", "test\r.pdf", true},
		{"very long filename", strings.Repeat("a", 300) + ".pdf", true},
		{"empty filename", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.validateFilename(tt.filename)

			if tt.expectError && err == nil {
				t.Errorf("Expected error for filename %s, but got none", tt.filename)
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error for filename %s: %v", tt.filename, err)
			}
		})
	}
}

func TestSecurityValidator_FileExtensionValidation(t *testing.T) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	tests := []struct {
		name        string
		filename    string
		mimeType    string
		expectError bool
	}{
		{"PDF with correct extension", "document.pdf", "application/pdf", false},
		{"JSON with correct extension", "data.json", "application/json", false},
		{"JPEG with jpg extension", "image.jpg", "image/jpeg", false},
		{"JPEG with jpeg extension", "image.jpeg", "image/jpeg", false},
		{"PNG with correct extension", "image.png", "image/png", false},
		{"PDF with wrong extension", "document.txt", "application/pdf", true},
		{"JSON with wrong extension", "data.pdf", "application/json", true},
		{"Image with wrong extension", "image.txt", "image/png", true},
		{"Text file without extension", "README", "text/plain", false},  // Allowed
		{"JSON without extension", "config", "application/json", false}, // Allowed
		{"PDF without extension", "document", "application/pdf", true},  // Not allowed
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.validateFileExtension(tt.filename, tt.mimeType)

			if tt.expectError && err == nil {
				t.Errorf("Expected error for %s/%s, but got none", tt.filename, tt.mimeType)
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error for %s/%s: %v", tt.filename, tt.mimeType, err)
			}
		})
	}
}

func TestSecurityValidator_MalwareDetection(t *testing.T) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	tests := []struct {
		name        string
		data        []byte
		mimeType    string
		expectError bool
	}{
		{"safe text file", []byte("Hello, World!"), "text/plain", false},
		{"EICAR test virus", eicarTestVirus, "text/plain", true},
		{"PE executable header", []byte("MZ\x90\x00\x03"), "text/plain", true},
		{"bash script", []byte("#!/bin/bash\necho 'hello'"), "text/plain", true},
		{"safe PDF", testPDFHeader, "application/pdf", false},
		{"malicious PDF", maliciousPDF, "application/pdf", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.scanFileContent(tt.data, tt.mimeType)

			if tt.expectError && err == nil {
				t.Errorf("Expected malware detection for %s, but got none", tt.name)
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected malware detection for %s: %v", tt.name, err)
			}
		})
	}
}

func TestSecurityValidator_JSONSecurity(t *testing.T) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	tests := []struct {
		name        string
		data        []byte
		expectError bool
	}{
		{"valid JSON", testJSONValid, false},
		{"prototype pollution", testJSONMalicious, true},
		{"deeply nested JSON", testJSONDeepNesting, true},
		{"simple nested JSON", []byte(`{"a":{"b":{"c":"value"}}}`), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.validateJSONSecurity(tt.data)

			if tt.expectError && err == nil {
				t.Errorf("Expected error for %s, but got none", tt.name)
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error for %s: %v", tt.name, err)
			}
		})
	}
}

func TestSecurityValidator_CSVSecurity(t *testing.T) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	tests := []struct {
		name        string
		data        []byte
		expectError bool
	}{
		{"safe CSV", testCSVSafe, false},
		{"CSV injection", testCSVMalicious, true},
		{"formula injection", []byte("name,value\n@SUM(1+1),test"), true},
		{"hyperlink injection", []byte("name,value\n=HYPERLINK(\"http://evil.com\"),test"), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.validateCSVSecurity(tt.data)

			if tt.expectError && err == nil {
				t.Errorf("Expected error for %s, but got none", tt.name)
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error for %s: %v", tt.name, err)
			}
		})
	}
}

func TestSecurityValidator_SizeValidation(t *testing.T) {
	config := &SecurityConfig{
		MaxImageSize:          1024, // 1KB
		MaxFileSize:           2048, // 2KB
		MaxTotalSize:          4096, // 4KB
		AllowedImageMIMEs:     []string{"image/png"},
		AllowedFileMIMEs:      []string{"text/plain"},
		EnableMalwareScanning: false, // Disable to test only size
		EnablePathValidation:  false,
		EnableContentScan:     false,
		CleanupInterval:       time.Second, // Positive interval to avoid panic
	}

	validator := NewSecurityValidator(config)
	defer validator.resourceMonitor.Shutdown()

	// Test image size limit
	largeImage := &ImageContent{
		Data:     make([]byte, 2048), // 2KB > 1KB limit
		MimeType: "image/png",
	}

	err := validator.ValidateContentSecurity(largeImage)
	if err == nil {
		t.Error("Expected size limit error for large image")
	}

	// Test file size limit
	largeFile := &FileContent{
		Data:     make([]byte, 4096), // 4KB > 2KB limit
		MimeType: "text/plain",
		Filename: "large.txt",
		FileSize: 4096,
	}

	err = validator.ValidateContentSecurity(largeFile)
	if err == nil {
		t.Error("Expected size limit error for large file")
	}

	// Test total message size limit using custom config
	largeMessage := &Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent(strings.Repeat("A", 5000)), // 5KB > 4KB limit
		},
	}

	// Use the validator with custom config for size validation
	err = validator.ValidateContentSecurity(largeMessage.Content[0])
	if err == nil {
		t.Error("Expected size limit error for large message content")
	}
}

func TestRateLimiter(t *testing.T) {
	limiter := NewRateLimiter(3) // 3 requests per minute

	clientID := "test-client"

	// Should allow first 3 requests
	for i := 0; i < 3; i++ {
		err := limiter.AllowRequest(clientID)
		if err != nil {
			t.Errorf("Request %d should be allowed: %v", i+1, err)
		}
	}

	// Should reject 4th request
	err := limiter.AllowRequest(clientID)
	if err == nil {
		t.Error("4th request should be rate limited")
	}

	// Test reset after time passes
	limiter.lastReset = time.Now().Add(-2 * time.Minute) // Simulate time passage
	err = limiter.AllowRequest(clientID)
	if err != nil {
		t.Errorf("Request should be allowed after reset: %v", err)
	}
}

func TestResourceMonitor(t *testing.T) {
	config := &SecurityConfig{
		MaxMemoryUsage:  1024, // 1KB
		CleanupInterval: 10 * time.Millisecond,
	}

	monitor := NewResourceMonitor(config)
	defer monitor.Shutdown()

	// Test memory tracking
	err := monitor.TrackMemoryUsage(512)
	if err != nil {
		t.Errorf("Should allow 512 bytes: %v", err)
	}

	err = monitor.TrackMemoryUsage(512)
	if err != nil {
		t.Errorf("Should allow total 1024 bytes: %v", err)
	}

	err = monitor.TrackMemoryUsage(1)
	if err == nil {
		t.Error("Should reject memory usage exceeding limit")
	}

	// Test file registration
	monitor.RegisterTemporaryFile("/tmp/test.txt")

	stats := monitor.GetResourceStats()
	if stats.TemporaryFiles != 1 {
		t.Errorf("Expected 1 temporary file, got %d", stats.TemporaryFiles)
	}

	// Test cleanup (files older than 1 hour)
	monitor.mu.Lock()
	monitor.temporaryFiles["/tmp/old.txt"] = time.Now().Add(-2 * time.Hour)
	monitor.mu.Unlock()

	cleaned := monitor.CleanupExpiredFiles()
	if cleaned != 1 {
		t.Errorf("Expected to clean 1 file, cleaned %d", cleaned)
	}
}

func TestSecurityAuditLogger(t *testing.T) {
	logger := NewSecurityAuditLogger()

	// Test content validation logging
	text := NewTextContent("test content")
	logger.LogContentValidation(text)

	// Test security event logging
	logger.LogSecurityEvent("TEST_EVENT", "Test message")

	// Test event retrieval
	events := logger.GetEventsByType("TEST_EVENT")
	if len(events) != 1 {
		t.Errorf("Expected 1 TEST_EVENT, got %d", len(events))
	}

	events = logger.GetEventsByType("CONTENT_VALIDATION")
	if len(events) != 1 {
		t.Errorf("Expected 1 CONTENT_VALIDATION event, got %d", len(events))
	}

	// Test recent events
	recentEvents := logger.GetRecentEvents(time.Now().Add(-1 * time.Minute))
	if len(recentEvents) != 2 {
		t.Errorf("Expected 2 recent events, got %d", len(recentEvents))
	}

	// Test total events
	total := logger.getTotalEvents()
	if total != 2 {
		t.Errorf("Expected 2 total events, got %d", total)
	}
}

func TestSecurityManager(t *testing.T) {
	config := &SecurityConfig{
		MaxRequestsPerMinute:  2,
		MaxImageSize:          1024,
		MaxFileSize:           2048,
		MaxTotalSize:          4096,
		AllowedImageMIMEs:     []string{"image/png"},
		AllowedFileMIMEs:      []string{"text/plain"},
		EnableMalwareScanning: true,
		EnablePathValidation:  true,
		EnableContentScan:     true,
	}

	manager := NewSecurityManager(config)
	defer manager.Shutdown()

	// Create test messages
	validMessage := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("Hello, World!"),
		},
	}

	invalidMessage := Message{
		Role: RoleUser,
		Content: []MessageContent{
			&ImageContent{
				Data:     testImagePNG,
				MimeType: "application/evil", // Not allowed
			},
		},
	}

	clientID := "test-client"

	// Test successful validation
	err := manager.ValidateRequest(clientID, []Message{validMessage})
	if err != nil {
		t.Errorf("Valid message should pass: %v", err)
	}

	// Test security validation failure
	err = manager.ValidateRequest(clientID, []Message{invalidMessage})
	if err == nil {
		t.Error("Invalid message should fail security validation")
	}

	// Test rate limiting (limit is 2 requests per minute)
	err = manager.ValidateRequest(clientID, []Message{validMessage})
	if err == nil {
		t.Error("Second request should be rate limited (limit is 2)")
	}

	// Test security stats
	stats := manager.GetSecurityStats()
	if stats.TotalValidations == 0 {
		t.Error("Should have logged validation events")
	}
	if stats.RateLimitHits == 0 {
		t.Error("Should have logged rate limit hits")
	}
}

func TestValidateMessageSecurity(t *testing.T) {
	tests := []struct {
		name        string
		message     *Message
		expectError bool
	}{
		{
			name: "valid message with text",
			message: &Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Hello, World!"),
				},
			},
			expectError: false,
		},
		{
			name: "valid message with image",
			message: &Message{
				Role: RoleUser,
				Content: []MessageContent{
					&ImageContent{
						Data:     testImagePNG,
						MimeType: "image/png",
					},
				},
			},
			expectError: false,
		},
		{
			name: "message with malicious content",
			message: &Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("<script>alert('xss')</script>"),
				},
			},
			expectError: true,
		},
		{
			name:        "nil message",
			message:     nil,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateMessageSecurity(tt.message)

			if tt.expectError && err == nil {
				t.Errorf("Expected error for %s, but got none", tt.name)
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error for %s: %v", tt.name, err)
			}
		})
	}
}

func TestSecurityValidator_EdgeCases(t *testing.T) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	// Test nil content
	err := validator.ValidateContentSecurity(nil)
	if err == nil {
		t.Error("Should reject nil content")
	}

	// Test zero-byte image
	emptyImage := &ImageContent{
		Data:     []byte{},
		MimeType: "image/png",
	}
	err = validator.ValidateContentSecurity(emptyImage)
	if err == nil {
		t.Error("Should reject empty image data")
	}

	// Test image too small for signature validation
	tinyImage := &ImageContent{
		Data:     []byte{0x01, 0x02}, // Only 2 bytes
		MimeType: "image/png",
	}
	err = validator.ValidateContentSecurity(tinyImage)
	if err == nil {
		t.Error("Should reject image too small for signature validation")
	}

	// Test file with null bytes claiming to be text
	fileWithNullBytes := &FileContent{
		Data:     []byte("Hello\x00World"),
		MimeType: "text/plain",
		Filename: "test.txt",
		FileSize: 11,
	}
	err = validator.ValidateContentSecurity(fileWithNullBytes)
	if err == nil {
		t.Error("Should reject text file with null bytes")
	}
}

func TestSecurityValidator_URLValidation(t *testing.T) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	tests := []struct {
		name        string
		url         string
		expectError bool
	}{
		{"valid HTTPS URL", "https://example.com/image.png", false},
		{"valid HTTP URL", "http://example.com/image.png", false},
		{"javascript scheme", "javascript:alert('xss')", true},
		{"data URI", "data:image/png;base64,iVBOR...", true},
		{"file scheme", "file:///etc/passwd", true},
		{"FTP scheme", "ftp://example.com/file.txt", true},
		{"localhost URL", "http://localhost/image.png", true},
		{"private IP", "https://192.168.1.1/image.png", true},
		{"path traversal", "https://example.com/../../../etc/passwd", true},
		{"URL encoded traversal", "https://example.com/%2e%2e/file", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.validateImageURL(tt.url)

			if tt.expectError && err == nil {
				t.Errorf("Expected error for URL %s, but got none", tt.url)
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error for URL %s: %v", tt.url, err)
			}
		})
	}
}

func TestSecurityValidator_OfficeDocumentScanning(t *testing.T) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	// Valid Office document (ZIP header)
	validOfficeDoc := []byte("PK\x03\x04\x14\x00\x06\x00")
	err := validator.scanOfficeContent(validOfficeDoc)
	if err != nil {
		t.Errorf("Valid office document should pass: %v", err)
	}

	// Invalid Office document (no ZIP header)
	invalidOfficeDoc := []byte("Not a ZIP file")
	err = validator.scanOfficeContent(invalidOfficeDoc)
	if err == nil {
		t.Error("Invalid office document should be rejected")
	}

	// Office document with macro indicators (should log but not fail)
	macroDoc := []byte("PK\x03\x04vbaProject.bin")
	err = validator.scanOfficeContent(macroDoc)
	if err != nil {
		t.Errorf("Office document with macros should not fail validation: %v", err)
	}

	// Check that macro detection was logged
	events := validator.GetAuditEventsByType("MACRO_DETECTED")
	if len(events) == 0 {
		t.Error("Macro detection should be logged")
	}
}

func TestValidateContentSecurity_PublicAPI(t *testing.T) {
	// Test the public API functions

	validContent := NewTextContent("Safe content")
	err := ValidateContentSecurity(validContent)
	if err != nil {
		t.Errorf("Valid content should pass: %v", err)
	}

	maliciousContent := NewTextContent("<script>alert('xss')</script>")
	err = ValidateContentSecurity(maliciousContent)
	if err == nil {
		t.Error("Malicious content should be rejected")
	}

	// Test with custom config
	strictConfig := &SecurityConfig{
		MaxTotalSize:      10, // Very small limit (smaller than "Safe content")
		EnableContentScan: true,
		AllowedImageMIMEs: []string{},  // No images allowed
		AllowedFileMIMEs:  []string{},  // No files allowed
		CleanupInterval:   time.Second, // Avoid panic
	}

	err = ValidateContentSecurityWithConfig(validContent, strictConfig)
	if err == nil {
		t.Error("Should fail with strict config due to size limit")
	}
}

// Helper function to generate deeply nested JSON for testing
func generateDeeplyNestedJSON(depth int) []byte {
	var buffer bytes.Buffer

	// Create opening braces
	for i := 0; i < depth; i++ {
		buffer.WriteString(`{"level":`)
	}

	buffer.WriteString(`"value"`)

	// Create closing braces
	for i := 0; i < depth; i++ {
		buffer.WriteString(`}`)
	}

	return buffer.Bytes()
}

// Benchmark tests for performance
func BenchmarkSecurityValidator_ValidateImageContent(b *testing.B) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	image := &ImageContent{
		Data:     testImagePNG,
		MimeType: "image/png",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = validator.ValidateContentSecurity(image)
	}
}

func BenchmarkSecurityValidator_ValidateFileContent(b *testing.B) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	file := &FileContent{
		Data:     []byte("Hello, World!"),
		MimeType: "text/plain",
		Filename: "test.txt",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = validator.ValidateContentSecurity(file)
	}
}

func BenchmarkSecurityValidator_MalwareScanning(b *testing.B) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	data := bytes.Repeat([]byte("safe content "), 1000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = validator.scanFileContent(data, "text/plain")
	}
}

func BenchmarkRateLimiter_AllowRequest(b *testing.B) {
	limiter := NewRateLimiter(1000000) // High limit to avoid blocking

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = limiter.AllowRequest("test-client")
	}
}

// Additional edge case tests
func TestSecurityValidator_ZeroDayAttacks(t *testing.T) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	// Test novel attack vectors
	tests := []struct {
		name        string
		content     MessageContent
		expectError bool
	}{
		{
			name: "polyglot file (PNG/JS)",
			content: &ImageContent{
				Data:     append(testImagePNG, []byte("/*PNG*/\nalert('polyglot')")...),
				MimeType: "image/png",
			},
			expectError: true, // Should detect JS injection
		},
		{
			name: "steganography indicator",
			content: &ImageContent{
				Data:     append(testImagePNG, []byte("HIDDEN_DATA_MARKER")...),
				MimeType: "image/png",
			},
			expectError: false, // Steganography detection would be advanced feature
		},
		{
			name:        "unicode normalization attack",
			content:     NewTextContent("file://\u202E\u0074\u0078\u0074\u002E"),
			expectError: false, // Basic implementation doesn't handle Unicode attacks
		},
		{
			name:        "HTML entity encoding bypass",
			content:     NewTextContent("&lt;script&gt;alert('xss')&lt;/script&gt;"),
			expectError: false, // Would need HTML entity decoding
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.ValidateContentSecurity(tt.content)

			if tt.expectError && err == nil {
				t.Errorf("Expected security detection for %s", tt.name)
			}
			if !tt.expectError && err != nil {
				// Log but don't fail - these are advanced attack vectors
				t.Logf("Advanced attack vector detected for %s: %v", tt.name, err)
			}
		})
	}
}

func TestSecurityValidator_FailureScenarios(t *testing.T) {
	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	// Test cleanup under various failure scenarios
	monitor := validator.resourceMonitor

	// Simulate resource leak
	for i := 0; i < 10; i++ {
		monitor.RegisterTemporaryFile(fmt.Sprintf("/tmp/file%d.tmp", i))
	}

	// Force cleanup
	cleaned := monitor.CleanupExpiredFiles()
	if cleaned != 0 {
		t.Logf("Cleaned %d files that were not expired", cleaned)
	}

	// Test memory tracking edge cases
	err := monitor.TrackMemoryUsage(0)
	if err != nil {
		t.Errorf("Should allow zero memory usage: %v", err)
	}

	monitor.ReleaseMemoryUsage(1000000) // Release more than tracked
	stats := monitor.GetResourceStats()
	if stats.TrackedMemory != 0 {
		t.Errorf("Memory usage should be reset to 0, got %d", stats.TrackedMemory)
	}
}
