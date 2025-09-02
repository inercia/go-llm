package llm

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"net/http"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"
)

// SecurityConfig defines configurable security policies
type SecurityConfig struct {
	// Content size limits (in bytes)
	MaxImageSize int64 `json:"max_image_size"`
	MaxFileSize  int64 `json:"max_file_size"`
	MaxTotalSize int64 `json:"max_total_size"`

	// MIME type restrictions
	AllowedImageMIMEs []string `json:"allowed_image_mimes"`
	AllowedFileMIMEs  []string `json:"allowed_file_mimes"`

	// Security policies
	EnableMalwareScanning bool `json:"enable_malware_scanning"`
	EnablePathValidation  bool `json:"enable_path_validation"`
	EnableContentScan     bool `json:"enable_content_scan"`

	// Rate limiting
	MaxRequestsPerMinute int           `json:"max_requests_per_minute"`
	MaxProcessingTime    time.Duration `json:"max_processing_time"`

	// Resource limits
	MaxMemoryUsage  int64         `json:"max_memory_usage"`
	CleanupInterval time.Duration `json:"cleanup_interval"`
}

// DefaultSecurityConfig returns a secure default configuration
func DefaultSecurityConfig() *SecurityConfig {
	return &SecurityConfig{
		MaxImageSize: 10 * 1024 * 1024,  // 10MB
		MaxFileSize:  50 * 1024 * 1024,  // 50MB
		MaxTotalSize: 100 * 1024 * 1024, // 100MB

		AllowedImageMIMEs: []string{
			"image/jpeg", "image/png", "image/gif", "image/webp",
			"image/bmp", "image/tiff", "image/svg+xml",
		},
		AllowedFileMIMEs: []string{
			"text/plain", "text/csv", "text/html", "text/markdown",
			"application/json", "application/pdf", "application/xml",
			"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
			"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
		},

		EnableMalwareScanning: true,
		EnablePathValidation:  true,
		EnableContentScan:     true,

		MaxRequestsPerMinute: 100,
		MaxProcessingTime:    30 * time.Second,

		MaxMemoryUsage:  500 * 1024 * 1024, // 500MB
		CleanupInterval: 5 * time.Minute,
	}
}

// SecurityValidator provides comprehensive content security validation
type SecurityValidator struct {
	config *SecurityConfig
	mu     sync.RWMutex

	// Rate limiting tracking
	requestCounts map[string]int
	lastReset     time.Time

	// Resource monitoring
	resourceMonitor *ResourceMonitor
	auditLogger     *SecurityAuditLogger
}

// NewSecurityValidator creates a new security validator with the given configuration
func NewSecurityValidator(config *SecurityConfig) *SecurityValidator {
	if config == nil {
		config = DefaultSecurityConfig()
	}

	return &SecurityValidator{
		config:          config,
		requestCounts:   make(map[string]int),
		lastReset:       time.Now(),
		resourceMonitor: NewResourceMonitor(config),
		auditLogger:     NewSecurityAuditLogger(),
	}
}

// ValidateContentSecurity performs comprehensive security validation on message content
func (sv *SecurityValidator) ValidateContentSecurity(content MessageContent) error {
	if content == nil {
		return errors.New("content cannot be nil")
	}

	// Log the validation attempt
	sv.auditLogger.LogContentValidation(content)

	// Basic content validation
	if err := content.Validate(); err != nil {
		sv.auditLogger.LogSecurityEvent("VALIDATION_FAILED", fmt.Sprintf("Basic validation failed: %v", err))
		return fmt.Errorf("content validation failed: %w", err)
	}

	// Type-specific security checks
	switch c := content.(type) {
	case *ImageContent:
		return sv.validateImageSecurity(c)
	case *FileContent:
		return sv.validateFileSecurity(c)
	case *TextContent:
		return sv.validateTextSecurity(c)
	default:
		sv.auditLogger.LogSecurityEvent("UNKNOWN_CONTENT_TYPE", fmt.Sprintf("Unknown content type: %T", content))
		return fmt.Errorf("unknown content type: %T", content)
	}
}

// validateImageSecurity validates image content for security threats
func (sv *SecurityValidator) validateImageSecurity(img *ImageContent) error {
	// Size validation
	if img.Size() > sv.config.MaxImageSize {
		err := fmt.Errorf("image size %d exceeds limit %d", img.Size(), sv.config.MaxImageSize)
		sv.auditLogger.LogSecurityEvent("SIZE_EXCEEDED", err.Error())
		return err
	}

	// MIME type validation
	if !sv.isAllowedImageMIME(img.MimeType) {
		err := fmt.Errorf("image MIME type %s not allowed", img.MimeType)
		sv.auditLogger.LogSecurityEvent("MIME_REJECTED", err.Error())
		return err
	}

	// Content signature validation for inline data
	if img.HasData() {
		if err := sv.validateImageSignature(img.Data, img.MimeType); err != nil {
			sv.auditLogger.LogSecurityEvent("SIGNATURE_MISMATCH", err.Error())
			return err
		}

		// Scan for malicious patterns
		if sv.config.EnableContentScan {
			if err := sv.scanImageContent(img.Data); err != nil {
				sv.auditLogger.LogSecurityEvent("MALICIOUS_CONTENT", err.Error())
				return err
			}
		}
	}

	// URL validation for external images
	if img.HasURL() {
		if err := sv.validateImageURL(img.URL); err != nil {
			sv.auditLogger.LogSecurityEvent("URL_REJECTED", err.Error())
			return err
		}
	}

	return nil
}

// validateFileSecurity validates file content for security threats
func (sv *SecurityValidator) validateFileSecurity(file *FileContent) error {
	// Size validation
	if file.Size() > sv.config.MaxFileSize {
		err := fmt.Errorf("file size %d exceeds limit %d", file.Size(), sv.config.MaxFileSize)
		sv.auditLogger.LogSecurityEvent("SIZE_EXCEEDED", err.Error())
		return err
	}

	// MIME type validation
	if !sv.isAllowedFileMIME(file.MimeType) {
		err := fmt.Errorf("file MIME type %s not allowed", file.MimeType)
		sv.auditLogger.LogSecurityEvent("MIME_REJECTED", err.Error())
		return err
	}

	// Filename security validation
	if sv.config.EnablePathValidation {
		if err := sv.validateFilename(file.Filename); err != nil {
			sv.auditLogger.LogSecurityEvent("PATH_TRAVERSAL", err.Error())
			return err
		}
	}

	// File extension consistency check
	if err := sv.validateFileExtension(file.Filename, file.MimeType); err != nil {
		sv.auditLogger.LogSecurityEvent("EXTENSION_MISMATCH", err.Error())
		return err
	}

	// Content validation for inline data
	if file.HasData() {
		// Binary signature validation
		if err := sv.validateFileSignature(file.Data, file.MimeType); err != nil {
			sv.auditLogger.LogSecurityEvent("SIGNATURE_MISMATCH", err.Error())
			return err
		}

		// Malware scanning
		if sv.config.EnableMalwareScanning {
			if err := sv.scanFileContent(file.Data, file.MimeType); err != nil {
				sv.auditLogger.LogSecurityEvent("MALWARE_DETECTED", err.Error())
				return err
			}
		}
	}

	// URL validation for external files
	if file.HasURL() {
		if err := sv.validateFileURL(file.URL); err != nil {
			sv.auditLogger.LogSecurityEvent("URL_REJECTED", err.Error())
			return err
		}
	}

	return nil
}

// validateTextSecurity validates text content for security threats
func (sv *SecurityValidator) validateTextSecurity(text *TextContent) error {
	// Size validation
	if text.Size() > sv.config.MaxTotalSize {
		err := fmt.Errorf("text size %d exceeds limit %d", text.Size(), sv.config.MaxTotalSize)
		sv.auditLogger.LogSecurityEvent("SIZE_EXCEEDED", err.Error())
		return err
	}

	// Content scanning for malicious patterns
	if sv.config.EnableContentScan {
		if err := sv.scanTextContent(text.Text); err != nil {
			sv.auditLogger.LogSecurityEvent("SUSPICIOUS_TEXT", err.Error())
			return err
		}
	}

	return nil
}

// GetAuditEventsByType returns security events of a specific type for testing purposes
func (sv *SecurityValidator) GetAuditEventsByType(eventType string) []SecurityEvent {
	return sv.auditLogger.GetEventsByType(eventType)
}

// isAllowedImageMIME checks if the image MIME type is in the allowlist
func (sv *SecurityValidator) isAllowedImageMIME(mimeType string) bool {
	sv.mu.RLock()
	defer sv.mu.RUnlock()

	for _, allowed := range sv.config.AllowedImageMIMEs {
		if mimeType == allowed {
			return true
		}
	}
	return false
}

// isAllowedFileMIME checks if the file MIME type is in the allowlist
func (sv *SecurityValidator) isAllowedFileMIME(mimeType string) bool {
	sv.mu.RLock()
	defer sv.mu.RUnlock()

	for _, allowed := range sv.config.AllowedFileMIMEs {
		if mimeType == allowed {
			return true
		}
	}
	return false
}

// validateImageSignature validates that the image data matches its declared MIME type
func (sv *SecurityValidator) validateImageSignature(data []byte, declaredMIME string) error {
	if len(data) < 12 {
		return errors.New("image data too small for signature validation")
	}

	// Detect actual MIME type from content
	detectedMIME := http.DetectContentType(data)

	// Special case: SVG files are often detected as text/plain by Go's http.DetectContentType
	if declaredMIME == "image/svg+xml" && strings.HasPrefix(detectedMIME, "text/plain") {
		if sv.detectSVGContent(data, declaredMIME) {
			// This is actually valid SVG content, allow it
			return nil
		}
	}

	// Handle specific cases where http.DetectContentType differs from expected
	actualMIME := sv.normalizeImageMIME(detectedMIME)
	expectedMIME := sv.normalizeImageMIME(declaredMIME)

	if actualMIME != expectedMIME {
		return fmt.Errorf("image signature mismatch: declared %s, detected %s", declaredMIME, detectedMIME)
	}

	return nil
}

// validateFileSignature validates that the file data matches its declared MIME type
func (sv *SecurityValidator) validateFileSignature(data []byte, declaredMIME string) error {
	if len(data) == 0 {
		return errors.New("empty file data")
	}

	// Special handling for text-based files
	if sv.isTextBasedMIME(declaredMIME) {
		return sv.validateTextFileSignature(data, declaredMIME)
	}

	// Binary file signature validation
	detectedMIME := http.DetectContentType(data)
	actualMIME := sv.normalizeFileMIME(detectedMIME)
	expectedMIME := sv.normalizeFileMIME(declaredMIME)

	if actualMIME != expectedMIME {
		return fmt.Errorf("file signature mismatch: declared %s, detected %s", declaredMIME, detectedMIME)
	}

	return nil
}

// validateFilename prevents path traversal and validates filename security
func (sv *SecurityValidator) validateFilename(filename string) error {
	if filename == "" {
		return errors.New("filename cannot be empty")
	}

	// Clean the path and check for traversal attempts
	cleaned := filepath.Clean(filename)
	if cleaned != filename {
		return fmt.Errorf("filename contains path traversal: %s", filename)
	}

	// Check for dangerous path components
	if strings.Contains(filename, "..") {
		return fmt.Errorf("filename contains parent directory reference: %s", filename)
	}

	// Check for absolute paths (Unix and Windows style)
	if filepath.IsAbs(filename) || (len(filename) >= 3 && filename[1] == ':' && (filename[2] == '\\' || filename[2] == '/')) {
		return fmt.Errorf("absolute paths not allowed: %s", filename)
	}

	// Check for dangerous characters
	dangerousChars := []string{"\x00", "\r", "\n", "\t"}
	for _, char := range dangerousChars {
		if strings.Contains(filename, char) {
			return fmt.Errorf("filename contains dangerous character: %s", filename)
		}
	}

	// Check filename length
	if len(filename) > 255 {
		return fmt.Errorf("filename too long: %d characters", len(filename))
	}

	return nil
}

// validateFileExtension ensures file extension matches MIME type
func (sv *SecurityValidator) validateFileExtension(filename, mimeType string) error {
	ext := strings.ToLower(filepath.Ext(filename))
	if ext == "" {
		// No extension - check if this is allowed for the MIME type
		if !sv.isExtensionlessAllowed(mimeType) {
			return fmt.Errorf("file extension required for MIME type %s", mimeType)
		}
		return nil
	}

	// Remove the dot from extension
	ext = strings.TrimPrefix(ext, ".")

	// Validate extension matches MIME type
	expectedExts := sv.getExpectedExtensions(mimeType)
	for _, expectedExt := range expectedExts {
		if ext == expectedExt {
			return nil
		}
	}

	return fmt.Errorf("file extension %s does not match MIME type %s", ext, mimeType)
}

// scanImageContent scans image data for malicious patterns
func (sv *SecurityValidator) scanImageContent(data []byte) error {
	// Check for embedded scripts in SVG
	if bytes.Contains(data, []byte("<script")) {
		return errors.New("image contains embedded script")
	}

	// Check for suspicious JavaScript patterns
	jsPatterns := [][]byte{
		[]byte("javascript:"),
		[]byte("eval("),
		[]byte("document."),
		[]byte("window."),
		[]byte("alert("), // Common in polyglot attacks and XSS
	}

	for _, pattern := range jsPatterns {
		if bytes.Contains(data, pattern) {
			return fmt.Errorf("image contains suspicious pattern: %s", string(pattern))
		}
	}

	// Check for HTML injection attempts
	htmlPatterns := [][]byte{
		[]byte("<iframe"),
		[]byte("<object"),
		[]byte("<embed"),
		[]byte("onload="),
		[]byte("onerror="),
	}

	for _, pattern := range htmlPatterns {
		if bytes.Contains(data, pattern) {
			return fmt.Errorf("image contains HTML injection pattern: %s", string(pattern))
		}
	}

	return nil
}

// scanFileContent scans file data for malicious patterns
func (sv *SecurityValidator) scanFileContent(data []byte, mimeType string) error {
	// Basic malware signatures (simplified - in production, use a real antivirus engine)
	malwareSignatures := [][]byte{
		// Common malware markers
		[]byte("X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"), // EICAR test
		[]byte("TVqQAAMAAAAEAAAA//8AALgAAAAAAAAAQAAAAAAAAAAA"),                          // PE executable header (base64)
		[]byte("MZ"),          // DOS/PE header
		[]byte("#!/bin/bash"), // Potential script
		[]byte("#!/bin/sh"),
		[]byte("cmd.exe"),
		[]byte("powershell"),
	}

	for _, signature := range malwareSignatures {
		if bytes.Contains(data, signature) {
			return fmt.Errorf("malware signature detected in file content")
		}
	}

	// PDF-specific security checks
	if mimeType == "application/pdf" {
		return sv.scanPDFContent(data)
	}

	// Office document security checks
	if sv.isOfficeDocument(mimeType) {
		return sv.scanOfficeContent(data)
	}

	return nil
}

// scanTextContent scans text for suspicious patterns
func (sv *SecurityValidator) scanTextContent(text string) error {
	// Check for injection patterns
	injectionPatterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)<script[^>]*>`),
		regexp.MustCompile(`(?i)javascript:`),
		regexp.MustCompile(`(?i)vbscript:`),
		regexp.MustCompile(`(?i)onload\s*=`),
		regexp.MustCompile(`(?i)onerror\s*=`),
		regexp.MustCompile(`(?i)eval\s*\(`),
		regexp.MustCompile(`(?i)document\.cookie`),
		regexp.MustCompile(`(?i)document\.location`),
	}

	for _, pattern := range injectionPatterns {
		if pattern.MatchString(text) {
			return fmt.Errorf("text contains suspicious pattern: %s", pattern.String())
		}
	}

	// Check for potential SQL injection patterns
	sqlPatterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)(\bunion\b.*\bselect\b)`),
		regexp.MustCompile(`(?i)(\bdrop\b.*\btable\b)`),
		regexp.MustCompile(`(?i)(\bdelete\b.*\bfrom\b)`),
		regexp.MustCompile(`(?i)(\binsert\b.*\binto\b)`),
		regexp.MustCompile(`(?i)(\bupdate\b.*\bset\b)`),
	}

	for _, pattern := range sqlPatterns {
		if pattern.MatchString(text) {
			sv.auditLogger.LogSecurityEvent("SQL_INJECTION_ATTEMPT", fmt.Sprintf("Pattern: %s", pattern.String()))
			// For text content, we log but don't necessarily block
		}
	}

	return nil
}

// scanPDFContent performs PDF-specific security scanning
func (sv *SecurityValidator) scanPDFContent(data []byte) error {
	// Check for PDF header
	if !bytes.HasPrefix(data, []byte("%PDF-")) {
		return errors.New("invalid PDF header")
	}

	// Check for suspicious PDF elements
	suspiciousPatterns := [][]byte{
		[]byte("/JavaScript"),
		[]byte("/JS"),
		[]byte("/OpenAction"),
		[]byte("/AA"), // Auto-Action
		[]byte("/Launch"),
		[]byte("/EmbeddedFile"),
	}

	for _, pattern := range suspiciousPatterns {
		if bytes.Contains(data, pattern) {
			return fmt.Errorf("PDF contains potentially dangerous element: %s", string(pattern))
		}
	}

	return nil
}

// scanOfficeContent performs Office document security scanning
func (sv *SecurityValidator) scanOfficeContent(data []byte) error {
	// Basic checks for Office documents (ZIP-based)
	if len(data) < 4 {
		return errors.New("office document too small")
	}

	// Check for ZIP header (Office docs are ZIP archives)
	zipHeaders := [][]byte{
		[]byte("PK\x03\x04"), // Standard ZIP
		[]byte("PK\x05\x06"), // Empty ZIP
		[]byte("PK\x07\x08"), // Spanned ZIP
	}

	hasZipHeader := false
	for _, header := range zipHeaders {
		if bytes.HasPrefix(data, header) {
			hasZipHeader = true
			break
		}
	}

	if !hasZipHeader {
		return errors.New("invalid Office document format")
	}

	// Check for suspicious macro patterns (basic detection)
	macroPatterns := [][]byte{
		[]byte("vbaProject.bin"),
		[]byte("macros/"),
		[]byte("Auto_Open"),
		[]byte("Workbook_Open"),
		[]byte("Document_Open"),
	}

	for _, pattern := range macroPatterns {
		if bytes.Contains(data, pattern) {
			sv.auditLogger.LogSecurityEvent("MACRO_DETECTED", fmt.Sprintf("Pattern: %s", string(pattern)))
			// Note: We log but don't necessarily block macros as they might be legitimate
		}
	}

	return nil
}

// validateImageURL validates external image URLs for security
func (sv *SecurityValidator) validateImageURL(url string) error {
	// Check for dangerous schemes
	if strings.HasPrefix(url, "javascript:") ||
		strings.HasPrefix(url, "data:") ||
		strings.HasPrefix(url, "file:") ||
		strings.HasPrefix(url, "ftp:") {
		return fmt.Errorf("dangerous URL scheme: %s", url)
	}

	// Only allow HTTPS and HTTP
	if !strings.HasPrefix(url, "https://") && !strings.HasPrefix(url, "http://") {
		return fmt.Errorf("only HTTP/HTTPS URLs allowed: %s", url)
	}

	// Check for suspicious patterns
	suspiciousPatterns := []string{
		"localhost", "127.0.0.1", "0.0.0.0",
		"169.254.", "192.168.", "10.", "172.16.", // Private IP ranges
		"../", "..\\", "%2e%2e", // Path traversal
	}

	for _, pattern := range suspiciousPatterns {
		if strings.Contains(strings.ToLower(url), pattern) {
			return fmt.Errorf("URL contains suspicious pattern: %s", pattern)
		}
	}

	return nil
}

// validateFileURL validates external file URLs for security
func (sv *SecurityValidator) validateFileURL(url string) error {
	// Use same validation as images but with additional restrictions
	if err := sv.validateImageURL(url); err != nil {
		return err
	}

	// Additional file-specific restrictions
	if strings.Contains(strings.ToLower(url), ".exe") ||
		strings.Contains(strings.ToLower(url), ".bat") ||
		strings.Contains(strings.ToLower(url), ".cmd") ||
		strings.Contains(strings.ToLower(url), ".scr") ||
		strings.Contains(strings.ToLower(url), ".com") {
		return fmt.Errorf("executable file URLs not allowed: %s", url)
	}

	return nil
}

// Helper methods for MIME type handling

func (sv *SecurityValidator) normalizeImageMIME(mimeType string) string {
	// Handle common variations
	switch mimeType {
	case "image/jpg":
		return "image/jpeg"
	default:
		return mimeType
	}
}

// Helper function to detect SVG content when http.DetectContentType returns text/plain
func (sv *SecurityValidator) detectSVGContent(data []byte, declaredMIME string) bool {
	if declaredMIME == "image/svg+xml" && len(data) > 0 {
		// Check if the content starts with SVG markers
		content := strings.TrimSpace(string(data))
		return strings.HasPrefix(content, "<svg") || strings.Contains(content, "<svg")
	}
	return false
}

func (sv *SecurityValidator) normalizeFileMIME(mimeType string) string {
	// Handle common variations and be more lenient for text files
	switch {
	case strings.HasPrefix(mimeType, "text/plain"):
		return "text/plain"
	default:
		return mimeType
	}
}

func (sv *SecurityValidator) isTextBasedMIME(mimeType string) bool {
	textMIMEs := []string{
		"text/plain", "text/csv", "text/html", "text/markdown",
		"application/json", "application/xml", "text/xml",
	}

	for _, textMIME := range textMIMEs {
		if mimeType == textMIME || strings.HasPrefix(mimeType, textMIME) {
			return true
		}
	}
	return false
}

func (sv *SecurityValidator) validateTextFileSignature(data []byte, declaredMIME string) error {
	// For text files, ensure content is valid UTF-8 and doesn't contain null bytes
	if bytes.Contains(data, []byte{0}) {
		return errors.New("text file contains null bytes")
	}

	// JSON-specific validation
	if declaredMIME == "application/json" {
		return sv.validateJSONSecurity(data)
	}

	// CSV-specific validation
	if declaredMIME == "text/csv" {
		return sv.validateCSVSecurity(data)
	}

	return nil
}

func (sv *SecurityValidator) validateJSONSecurity(data []byte) error {
	// Check for excessively deep nesting (potential DoS)
	nestingLevel := 0
	maxNesting := 32

	for _, b := range data {
		switch b {
		case '{', '[':
			nestingLevel++
			if nestingLevel > maxNesting {
				return fmt.Errorf("JSON nesting too deep: %d levels", nestingLevel)
			}
		case '}', ']':
			nestingLevel--
		}
	}

	// Check for suspicious JSON patterns
	if bytes.Contains(data, []byte(`"__proto__"`)) {
		return errors.New("JSON contains prototype pollution attempt")
	}

	return nil
}

func (sv *SecurityValidator) validateCSVSecurity(data []byte) error {
	// Check for CSV injection attempts
	csvInjectionPatterns := [][]byte{
		[]byte("=cmd|"),
		[]byte("=system|"),
		[]byte("@SUM("),
		[]byte("=HYPERLINK("),
		[]byte("+cmd|"),
		[]byte("-cmd|"),
	}

	for _, pattern := range csvInjectionPatterns {
		if bytes.Contains(data, pattern) {
			return fmt.Errorf("CSV contains injection pattern: %s", string(pattern))
		}
	}

	return nil
}

func (sv *SecurityValidator) isOfficeDocument(mimeType string) bool {
	officeMIMEs := []string{
		"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
		"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
		"application/vnd.openxmlformats-officedocument.presentationml.presentation",
	}

	for _, officeMIME := range officeMIMEs {
		if mimeType == officeMIME {
			return true
		}
	}
	return false
}

func (sv *SecurityValidator) isExtensionlessAllowed(mimeType string) bool {
	// Some MIME types don't require extensions
	extensionlessAllowed := []string{
		"text/plain",
		"application/json",
	}

	for _, allowed := range extensionlessAllowed {
		if mimeType == allowed {
			return true
		}
	}
	return false
}

func (sv *SecurityValidator) getExpectedExtensions(mimeType string) []string {
	extensions := map[string][]string{
		"image/jpeg":    {"jpg", "jpeg"},
		"image/png":     {"png"},
		"image/gif":     {"gif"},
		"image/webp":    {"webp"},
		"image/bmp":     {"bmp"},
		"image/tiff":    {"tiff", "tif"},
		"image/svg+xml": {"svg"},

		"text/plain":       {"txt", "text"},
		"text/csv":         {"csv"},
		"text/html":        {"html", "htm"},
		"text/markdown":    {"md", "markdown"},
		"application/json": {"json"},
		"application/pdf":  {"pdf"},
		"application/xml":  {"xml"},

		"application/vnd.openxmlformats-officedocument.wordprocessingml.document": {"docx"},
		"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":       {"xlsx"},
	}

	if exts, found := extensions[mimeType]; found {
		return exts
	}
	return []string{}
}

// ResourceMonitor tracks resource usage for security and cleanup
type ResourceMonitor struct {
	config         *SecurityConfig
	mu             sync.RWMutex
	temporaryFiles map[string]time.Time
	memoryUsage    int64
	processedCount int64
	lastCleanup    time.Time
	cleanupTicker  *time.Ticker
	shutdownChan   chan bool
}

// NewResourceMonitor creates a new resource monitor
func NewResourceMonitor(config *SecurityConfig) *ResourceMonitor {
	rm := &ResourceMonitor{
		config:         config,
		temporaryFiles: make(map[string]time.Time),
		lastCleanup:    time.Now(),
		shutdownChan:   make(chan bool, 1),
	}

	// Start cleanup goroutine if interval is positive
	if config.CleanupInterval > 0 {
		rm.cleanupTicker = time.NewTicker(config.CleanupInterval)
		go rm.cleanupWorker()
	}

	return rm
}

// RegisterTemporaryFile registers a temporary file for cleanup tracking
func (rm *ResourceMonitor) RegisterTemporaryFile(filepath string) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rm.temporaryFiles[filepath] = time.Now()
}

// TrackMemoryUsage updates memory usage tracking
func (rm *ResourceMonitor) TrackMemoryUsage(size int64) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rm.memoryUsage += size
	if rm.memoryUsage > rm.config.MaxMemoryUsage {
		return fmt.Errorf("memory usage %d exceeds limit %d", rm.memoryUsage, rm.config.MaxMemoryUsage)
	}

	return nil
}

// ReleaseMemoryUsage decreases memory usage tracking
func (rm *ResourceMonitor) ReleaseMemoryUsage(size int64) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rm.memoryUsage -= size
	if rm.memoryUsage < 0 {
		rm.memoryUsage = 0
	}
}

// GetResourceStats returns current resource usage statistics
func (rm *ResourceMonitor) GetResourceStats() ResourceStats {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return ResourceStats{
		TemporaryFiles: len(rm.temporaryFiles),
		TrackedMemory:  rm.memoryUsage,
		SystemMemory:   safeUint64ToInt64(m.Sys),
		HeapMemory:     safeUint64ToInt64(m.HeapSys),
		ProcessedCount: rm.processedCount,
		LastCleanup:    rm.lastCleanup,
	}
}

// CleanupExpiredFiles removes expired temporary files
func (rm *ResourceMonitor) CleanupExpiredFiles() int {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	now := time.Now()
	cleanupAge := 1 * time.Hour // Files older than 1 hour

	cleaned := 0
	for filepath, createdAt := range rm.temporaryFiles {
		if now.Sub(createdAt) > cleanupAge {
			// Note: In a real implementation, you'd actually delete the file here
			// For this implementation, we're tracking for monitoring purposes
			delete(rm.temporaryFiles, filepath)
			cleaned++
		}
	}

	rm.lastCleanup = now
	return cleaned
}

// Shutdown stops the resource monitor
func (rm *ResourceMonitor) Shutdown() {
	rm.shutdownChan <- true
	if rm.cleanupTicker != nil {
		rm.cleanupTicker.Stop()
	}
}

// cleanupWorker runs periodic cleanup operations
func (rm *ResourceMonitor) cleanupWorker() {
	for {
		select {
		case <-rm.cleanupTicker.C:
			cleaned := rm.CleanupExpiredFiles()
			if cleaned > 0 {
				// Log cleanup activity (in real impl, use proper logging)
				_ = cleaned
			}
		case <-rm.shutdownChan:
			return
		}
	}
}

// ResourceStats contains resource usage statistics
type ResourceStats struct {
	TemporaryFiles int       `json:"temporary_files"`
	TrackedMemory  int64     `json:"tracked_memory"`
	SystemMemory   int64     `json:"system_memory"`
	HeapMemory     int64     `json:"heap_memory"`
	ProcessedCount int64     `json:"processed_count"`
	LastCleanup    time.Time `json:"last_cleanup"`
}

// SecurityAuditLogger handles security event logging and monitoring
type SecurityAuditLogger struct {
	mu     sync.RWMutex
	events []SecurityEvent
}

// NewSecurityAuditLogger creates a new security audit logger
func NewSecurityAuditLogger() *SecurityAuditLogger {
	return &SecurityAuditLogger{
		events: make([]SecurityEvent, 0),
	}
}

// LogContentValidation logs a content validation event
func (sal *SecurityAuditLogger) LogContentValidation(content MessageContent) {
	event := SecurityEvent{
		Timestamp:   time.Now(),
		EventType:   "CONTENT_VALIDATION",
		Message:     fmt.Sprintf("Validating %s content (size: %d bytes)", content.Type(), content.Size()),
		ContentHash: sal.generateContentHash(content),
	}

	sal.mu.Lock()
	sal.events = append(sal.events, event)
	sal.mu.Unlock()
}

// LogSecurityEvent logs a security-related event
func (sal *SecurityAuditLogger) LogSecurityEvent(eventType, message string) {
	event := SecurityEvent{
		Timestamp: time.Now(),
		EventType: eventType,
		Message:   message,
	}

	sal.mu.Lock()
	sal.events = append(sal.events, event)
	sal.mu.Unlock()
}

// GetRecentEvents returns recent security events
func (sal *SecurityAuditLogger) GetRecentEvents(since time.Time) []SecurityEvent {
	sal.mu.RLock()
	defer sal.mu.RUnlock()

	var recentEvents []SecurityEvent
	for _, event := range sal.events {
		if event.Timestamp.After(since) {
			recentEvents = append(recentEvents, event)
		}
	}

	return recentEvents
}

// GetEventsByType returns events of a specific type
func (sal *SecurityAuditLogger) GetEventsByType(eventType string) []SecurityEvent {
	sal.mu.RLock()
	defer sal.mu.RUnlock()

	var filteredEvents []SecurityEvent
	for _, event := range sal.events {
		if event.EventType == eventType {
			filteredEvents = append(filteredEvents, event)
		}
	}

	return filteredEvents
}

// SecurityEvent represents a security-related event for auditing
type SecurityEvent struct {
	Timestamp   time.Time `json:"timestamp"`
	EventType   string    `json:"event_type"`
	Message     string    `json:"message"`
	ContentHash string    `json:"content_hash,omitempty"`
}

// generateContentHash creates a hash for content identification (without exposing content)
func (sal *SecurityAuditLogger) generateContentHash(content MessageContent) string {
	var data []byte

	switch c := content.(type) {
	case *ImageContent:
		if c.HasData() {
			data = c.Data
		} else {
			data = []byte(c.URL)
		}
	case *FileContent:
		if c.HasData() {
			data = c.Data
		} else {
			data = []byte(c.URL)
		}
	case *TextContent:
		data = []byte(c.Text)
	}

	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash[:8]) // First 8 bytes for identification
}

// ValidateContentSecurity is the main entry point for content security validation
func ValidateContentSecurity(content MessageContent) error {
	validator := NewSecurityValidator(nil) // Use default config
	defer validator.resourceMonitor.Shutdown()

	return validator.ValidateContentSecurity(content)
}

// ValidateContentSecurityWithConfig validates content with custom security config
func ValidateContentSecurityWithConfig(content MessageContent, config *SecurityConfig) error {
	validator := NewSecurityValidator(config)
	defer validator.resourceMonitor.Shutdown()

	return validator.ValidateContentSecurity(content)
}

// ValidateMessageSecurity validates all content in a message for security threats
func ValidateMessageSecurity(message *Message) error {
	if message == nil {
		return errors.New("message cannot be nil")
	}

	validator := NewSecurityValidator(nil)
	defer validator.resourceMonitor.Shutdown()

	totalSize := message.TotalSize()
	if totalSize > validator.config.MaxTotalSize {
		err := fmt.Errorf("message total size %d exceeds limit %d", totalSize, validator.config.MaxTotalSize)
		validator.auditLogger.LogSecurityEvent("TOTAL_SIZE_EXCEEDED", err.Error())
		return err
	}

	// Validate each content item
	for i, content := range message.Content {
		if err := validator.ValidateContentSecurity(content); err != nil {
			return fmt.Errorf("content item %d failed security validation: %w", i, err)
		}
	}

	return nil
}

// RateLimiter provides rate limiting functionality for security
type RateLimiter struct {
	mu            sync.RWMutex
	requestCounts map[string]int
	lastReset     time.Time
	limit         int
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(requestsPerMinute int) *RateLimiter {
	return &RateLimiter{
		requestCounts: make(map[string]int),
		lastReset:     time.Now(),
		limit:         requestsPerMinute,
	}
}

// AllowRequest checks if a request should be allowed based on rate limiting
func (rl *RateLimiter) AllowRequest(clientID string) error {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()

	// Reset counts every minute
	if now.Sub(rl.lastReset) >= time.Minute {
		rl.requestCounts = make(map[string]int)
		rl.lastReset = now
	}

	// Check current count
	count := rl.requestCounts[clientID]
	if count >= rl.limit {
		return fmt.Errorf("rate limit exceeded for client %s: %d requests/minute", clientID, count)
	}

	// Increment count
	rl.requestCounts[clientID] = count + 1

	return nil
}

// SecurityManager provides centralized security management
type SecurityManager struct {
	validator   *SecurityValidator
	rateLimiter *RateLimiter
	config      *SecurityConfig
}

// NewSecurityManager creates a new security manager with comprehensive security controls
func NewSecurityManager(config *SecurityConfig) *SecurityManager {
	if config == nil {
		config = DefaultSecurityConfig()
	}

	return &SecurityManager{
		validator:   NewSecurityValidator(config),
		rateLimiter: NewRateLimiter(config.MaxRequestsPerMinute),
		config:      config,
	}
}

// ValidateRequest performs comprehensive request validation
func (sm *SecurityManager) ValidateRequest(clientID string, messages []Message) error {
	// Rate limiting
	if err := sm.rateLimiter.AllowRequest(clientID); err != nil {
		sm.validator.auditLogger.LogSecurityEvent("RATE_LIMIT_EXCEEDED", err.Error())
		return err
	}

	// Content security validation
	for i, message := range messages {
		if err := ValidateMessageSecurity(&message); err != nil {
			return fmt.Errorf("message %d failed security validation: %w", i, err)
		}
	}

	return nil
}

// GetSecurityStats returns current security statistics
func (sm *SecurityManager) GetSecurityStats() SecurityStats {
	resourceStats := sm.validator.resourceMonitor.GetResourceStats()

	return SecurityStats{
		ResourceStats:      resourceStats,
		TotalValidations:   sm.validator.auditLogger.getTotalEvents(),
		SecurityViolations: len(sm.validator.auditLogger.GetEventsByType("MALICIOUS_CONTENT")),
		RateLimitHits:      len(sm.validator.auditLogger.GetEventsByType("RATE_LIMIT_EXCEEDED")),
	}
}

// Shutdown gracefully shuts down the security manager
func (sm *SecurityManager) Shutdown() {
	if sm.validator != nil && sm.validator.resourceMonitor != nil {
		sm.validator.resourceMonitor.Shutdown()
	}
}

// SecurityStats contains comprehensive security statistics
type SecurityStats struct {
	ResourceStats      ResourceStats `json:"resource_stats"`
	TotalValidations   int           `json:"total_validations"`
	SecurityViolations int           `json:"security_violations"`
	RateLimitHits      int           `json:"rate_limit_hits"`
}

// getTotalEvents returns the total number of logged events
func (sal *SecurityAuditLogger) getTotalEvents() int {
	sal.mu.RLock()
	defer sal.mu.RUnlock()
	return len(sal.events)
}

// safeUint64ToInt64 safely converts uint64 to int64, preventing overflow
func safeUint64ToInt64(val uint64) int64 {
	if val > math.MaxInt64 {
		return math.MaxInt64
	}
	return int64(val)
}
