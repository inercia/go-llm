// Package llm provides retry functionality for chat completions with exponential backoff.
//
// Examples:
//
// Basic usage with default configuration (3 retries, 1s base delay, 2x backoff):
//
//	client, _ := llm.NewOpenAIClient(config)
//	retryClient := llm.RetryChatCompletion(client)
//	resp, err := retryClient.ChatCompletion(ctx, request)
//
// Custom retry configuration for high-throughput scenarios:
//
//	retryConfig := llm.RetryConfig{
//		MaxRetries:    10,
//		BaseDelay:     100 * time.Millisecond,
//		BackoffFactor: 1.2,
//		Jitter:        true,
//	}
//	retryClient := llm.RetryChatCompletion(client, retryConfig)
//
// Conservative retry for rate-limited APIs:
//
//	retryConfig := llm.RetryConfig{
//		MaxRetries:    5,
//		BaseDelay:     2 * time.Second,
//		MaxDelay:      5 * time.Minute,
//		BackoffFactor: 2.5,
//		Jitter:        true,
//	}
//	retryClient := llm.RetryChatCompletion(client, retryConfig)
//
// Custom retryable errors:
//
//	retryConfig := llm.RetryConfig{
//		RetryableErrors: []string{"rate_limit_exceeded", "quota_exceeded", "temporary_unavailable"},
//	}
//	retryClient := llm.RetryChatCompletion(client, retryConfig)
package llm

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"math"
	"time"
)

// secureRandomFloat64 generates a cryptographically secure random float64 between 0 and 1
func secureRandomFloat64() (float64, error) {
	var bytes [8]byte
	_, err := rand.Read(bytes[:])
	if err != nil {
		return 0, err
	}
	// Convert bytes to uint64, then to float64 between 0 and 1
	return float64(binary.BigEndian.Uint64(bytes[:])) / float64(^uint64(0)), nil
}

// ChatCompleter defines an interface for any client that can perform chat completions.
// Any client implementing this interface can be wrapped with RetryChatCompletion.
//
// This interface is implemented by all built-in clients:
//   - OpenAIClient
//   - OpenRouterClient
//   - GeminiClient
//   - OllamaClient
//   - MockClient
//
// Custom clients can also implement this interface:
//
//	type MyCustomClient struct { ... }
//	func (c *MyCustomClient) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
//		// Your implementation
//	}
//
//	// Then wrap with retry:
//	retryClient := llm.RetryChatCompletion(&MyCustomClient{...})
type ChatCompleter interface {
	ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error)
}

// RetryConfig defines configuration options for the retry mechanism.
//
// Examples of different retry strategies:
//
// Web application (balance speed and reliability):
//
//	RetryConfig{MaxRetries: 3, BaseDelay: 1*time.Second, BackoffFactor: 2.0}
//
// Batch processing (aggressive retries):
//
//	RetryConfig{MaxRetries: 10, BaseDelay: 100*time.Millisecond, BackoffFactor: 1.2}
//
// Real-time system (quick failures):
//
//	RetryConfig{MaxRetries: 1, BaseDelay: 200*time.Millisecond, BackoffFactor: 2.0}
//
// Rate-limited API (respectful backoff):
//
//	RetryConfig{MaxRetries: 5, BaseDelay: 5*time.Second, MaxDelay: 10*time.Minute, BackoffFactor: 2.5}
//
// Integration tests (only retry rate limits, not server errors):
//
//	RetryConfig{MaxRetries: 3, BaseDelay: 2*time.Second, RetryOnStatusCodes: []int{429}}
//
// Production API (retry specific errors):
//
//	RetryConfig{MaxRetries: 5, BaseDelay: 1*time.Second,
//	  RetryOnStatusCodes: []int{429, 502, 503},
//	  RetryOnErrorTypes: []string{"rate_limit_error", "temporary_error"}}
type RetryConfig struct {
	// MaxRetries is the maximum number of retry attempts (default: 3).
	// Total requests = MaxRetries + 1 (original attempt).
	// Example: MaxRetries: 5 allows up to 6 total requests.
	MaxRetries int

	// BaseDelay is the initial delay between retries (default: 1 second).
	// Each retry multiplies this by BackoffFactor.
	// Example: BaseDelay: 500*time.Millisecond starts with 500ms delay.
	BaseDelay time.Duration

	// MaxDelay caps the maximum delay between retries (default: 60 seconds).
	// Prevents exponential backoff from becoming too large.
	// Example: MaxDelay: 30*time.Second limits delays to 30 seconds.
	MaxDelay time.Duration

	// BackoffFactor multiplies the delay after each retry (default: 2.0).
	// Higher values increase delays more rapidly.
	// Example: BackoffFactor: 1.5 = gentler increase, 3.0 = aggressive increase.
	BackoffFactor float64

	// Jitter adds randomness to delays to prevent thundering herd (default: true).
	// Multiplies delay by random factor between 0.5 and 1.5.
	// Example: Jitter: false for predictable timing in tests.
	Jitter bool

	// RetryableErrors lists additional error codes that should trigger retries.
	// Used when RetryOnErrorTypes is empty (for backward compatibility).
	// Example: []string{"quota_exceeded", "temporary_unavailable"}
	RetryableErrors []string

	// RetryOnStatusCodes specifies exact HTTP status codes to retry on.
	// If empty, uses default behavior (429, 5xx). If specified, ONLY these codes trigger retries.
	// Example: []int{429} retries only on rate limiting
	// Example: []int{429, 502, 503, 504} retries on rate limits and specific server errors
	RetryOnStatusCodes []int

	// RetryOnErrorTypes specifies exact error types to retry on.
	// If empty, uses default behavior ("rate_limit_error"). If specified, ONLY these types trigger retries.
	// Example: []string{"rate_limit_error"} retries only on rate limit errors
	// Example: []string{"rate_limit_error", "api_error"} retries on rate limits and API errors
	RetryOnErrorTypes []string
}

// DefaultRetryConfig returns a sensible default retry configuration
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:      3,
		BaseDelay:       1 * time.Second,
		MaxDelay:        60 * time.Second,
		BackoffFactor:   2.0,
		Jitter:          true,
		RetryableErrors: []string{"rate_limit_exceeded"},
	}
}

// RetryableChatCompleter wraps a ChatCompleter with retry functionality
type RetryableChatCompleter struct {
	client ChatCompleter
	config RetryConfig
}

// RetryChatCompletion creates a new retryable wrapper around any ChatCompleter.
// It automatically retries requests when throttling errors (HTTP 429), rate limit errors,
// or temporary server errors (5xx) occur, using exponential backoff with optional jitter.
//
// Examples:
//
// Default configuration (recommended for most cases):
//
//	retryClient := llm.RetryChatCompletion(client)
//
// With any LLM client:
//
//	openaiClient, _ := llm.NewOpenAIClient(openaiConfig)
//	retryOpenAI := llm.RetryChatCompletion(openaiClient)
//
//	openrouterClient, _ := llm.NewOpenRouterClient(openrouterConfig)
//	retryOpenRouter := llm.RetryChatCompletion(openrouterClient)
//
//	geminiClient, _ := llm.NewGeminiClient(geminiConfig)
//	retryGemini := llm.RetryChatCompletion(geminiClient)
//
// Aggressive retry for high-throughput production systems:
//
//	aggressiveConfig := llm.RetryConfig{
//		MaxRetries:    10,                    // More retries
//		BaseDelay:     50 * time.Millisecond, // Start small
//		BackoffFactor: 1.1,                   // Gradual increase
//		Jitter:        true,                  // Prevent thundering herd
//	}
//	retryClient := llm.RetryChatCompletion(client, aggressiveConfig)
//
// Conservative retry for development/testing:
//
//	conservativeConfig := llm.RetryConfig{
//		MaxRetries:    2,               // Fewer retries
//		BaseDelay:     3 * time.Second, // Longer delays
//		BackoffFactor: 3.0,             // Rapid increase
//		Jitter:        false,           // Predictable timing
//	}
//	retryClient := llm.RetryChatCompletion(client, conservativeConfig)
//
// Usage with context timeout (recommended):
//
//	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
//	defer cancel()
//	resp, err := retryClient.ChatCompletion(ctx, request)
//
// Integration tests (only retry on rate limits, not server errors):
//
//	testConfig := llm.RetryConfig{
//		MaxRetries:         3,
//		BaseDelay:          2 * time.Second,
//		RetryOnStatusCodes: []int{429}, // Only retry HTTP 429
//	}
//	retryClient := llm.RetryChatCompletion(client, testConfig)
//
// Production (retry specific combinations):
//
//	prodConfig := llm.RetryConfig{
//		MaxRetries:         5,
//		BaseDelay:          1 * time.Second,
//		RetryOnStatusCodes: []int{429, 502, 503},          // Rate limits + specific server errors
//		RetryOnErrorTypes:  []string{"rate_limit_error"},  // Plus rate limit error types
//	}
//	retryClient := llm.RetryChatCompletion(client, prodConfig)
func RetryChatCompletion(client ChatCompleter, config ...RetryConfig) ChatCompleter {
	cfg := DefaultRetryConfig()
	if len(config) > 0 {
		cfg = config[0]
		// Ensure sane defaults for zero values
		if cfg.MaxRetries <= 0 {
			cfg.MaxRetries = 3
		}
		if cfg.BaseDelay <= 0 {
			cfg.BaseDelay = 1 * time.Second
		}
		if cfg.MaxDelay <= 0 {
			cfg.MaxDelay = 60 * time.Second
		}
		if cfg.BackoffFactor <= 0 {
			cfg.BackoffFactor = 2.0
		}
		if cfg.RetryableErrors == nil {
			cfg.RetryableErrors = []string{"rate_limit_exceeded"}
		}
	}

	return &RetryableChatCompleter{
		client: client,
		config: cfg,
	}
}

// ChatCompletion executes the chat completion with retry logic
func (r *RetryableChatCompleter) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	var lastErr error

	for attempt := 0; attempt <= r.config.MaxRetries; attempt++ {
		// Make the request
		resp, err := r.client.ChatCompletion(ctx, req)
		if err == nil {
			return resp, nil
		}

		lastErr = err

		// Don't retry on the last attempt
		if attempt == r.config.MaxRetries {
			break
		}

		// Check if this error should trigger a retry
		if !r.isRetryableError(err) {
			return nil, err
		}

		// Calculate delay for this attempt
		delay := r.calculateDelay(attempt)

		// Create a timer for the delay, but also respect context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
			// Continue with retry
		}
	}

	return nil, lastErr
}

// isRetryableError determines if an error should trigger a retry
func (r *RetryableChatCompleter) isRetryableError(err error) bool {
	// Check if it's our standardized Error type
	llmErr, ok := err.(*Error)
	if !ok {
		return false
	}

	// If specific status codes are configured, only retry on those
	if len(r.config.RetryOnStatusCodes) > 0 {
		for _, code := range r.config.RetryOnStatusCodes {
			if llmErr.StatusCode == code {
				return true
			}
		}
		// If status codes are specified but this error doesn't match, don't check other conditions
		// unless error types are also specified (both can be used together)
		if len(r.config.RetryOnErrorTypes) == 0 {
			return false
		}
	}

	// If specific error types are configured, only retry on those
	if len(r.config.RetryOnErrorTypes) > 0 {
		for _, errorType := range r.config.RetryOnErrorTypes {
			if llmErr.Type == errorType {
				return true
			}
		}
		// If error types are specified but this error doesn't match, don't check other conditions
		// unless status codes matched above
		if len(r.config.RetryOnStatusCodes) == 0 || !r.statusCodeMatches(llmErr.StatusCode) {
			return false
		}
	}

	// If neither RetryOnStatusCodes nor RetryOnErrorTypes are specified,
	// use backward-compatible default behavior
	if len(r.config.RetryOnStatusCodes) == 0 && len(r.config.RetryOnErrorTypes) == 0 {
		// Check for rate limit errors (backward compatibility)
		if llmErr.Type == "rate_limit_error" {
			return true
		}

		// Check for specific error codes we want to retry (backward compatibility)
		for _, retryableCode := range r.config.RetryableErrors {
			if llmErr.Code == retryableCode {
				return true
			}
		}

		// Check for HTTP 429 status code (backward compatibility)
		if llmErr.StatusCode == 429 {
			return true
		}

		// Also retry on server errors (5xx) as they might be temporary (backward compatibility)
		if llmErr.StatusCode >= 500 && llmErr.StatusCode < 600 {
			return true
		}
	}

	return false
}

// statusCodeMatches checks if the status code matches any in RetryOnStatusCodes
func (r *RetryableChatCompleter) statusCodeMatches(statusCode int) bool {
	for _, code := range r.config.RetryOnStatusCodes {
		if statusCode == code {
			return true
		}
	}
	return false
}

// calculateDelay computes the delay for a given retry attempt using exponential backoff
func (r *RetryableChatCompleter) calculateDelay(attempt int) time.Duration {
	// Calculate exponential backoff: baseDelay * (backoffFactor ^ attempt)
	delay := float64(r.config.BaseDelay) * math.Pow(r.config.BackoffFactor, float64(attempt))

	// Apply jitter if enabled (random factor between 0.5 and 1.5)
	if r.config.Jitter {
		randomValue, err := secureRandomFloat64()
		if err != nil {
			// If we can't generate secure random, use maximum jitter factor for safety
			randomValue = 1.0
		}
		jitterFactor := 0.5 + randomValue // Random value between 0.5 and 1.5
		delay *= jitterFactor
	}

	// Cap at maximum delay
	if delay > float64(r.config.MaxDelay) {
		delay = float64(r.config.MaxDelay)
	}

	return time.Duration(delay)
}

// Ensure RetryableChatCompleter implements ChatCompleter
var _ ChatCompleter = (*RetryableChatCompleter)(nil)

// Example usage patterns:
//
// 1. Quick start with any existing client:
//
//	client, _ := llm.NewOpenAIClient(llm.ClientConfig{...})
//	retryClient := llm.RetryChatCompletion(client)
//	response, err := retryClient.ChatCompletion(ctx, request)
//
// 2. Production setup with custom config:
//
//	config := llm.RetryConfig{
//		MaxRetries:    5,
//		BaseDelay:     2 * time.Second,
//		MaxDelay:      2 * time.Minute,
//		BackoffFactor: 2.0,
//		Jitter:        true,
//	}
//	retryClient := llm.RetryChatCompletion(client, config)
//
//	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
//	defer cancel()
//
//	request := llm.ChatRequest{
//		Model: "gpt-3.5-turbo",
//		Messages: []llm.Message{
//			llm.NewTextMessage(llm.RoleUser, "Hello!"),
//		},
//	}
//
//	response, err := retryClient.ChatCompletion(ctx, request)
//	if err != nil {
//		// Handle error after all retries exhausted
//		// Handle error (add "log" import if using log.Printf)
//		log.Printf("Failed after retries: %v", err)
//	} else {
//		// Success! (add "log" import if using log.Printf)
//		log.Printf("Response: %s", response.Choices[0].Message.GetText())
//	}
//
// 3. Different strategies for different scenarios:
//
//	// High-frequency trading bot (fail fast)
//	fastFail := llm.RetryConfig{MaxRetries: 1, BaseDelay: 10*time.Millisecond}
//
//	// Batch job (persistent)
//	persistent := llm.RetryConfig{MaxRetries: 10, BaseDelay: 5*time.Second, MaxDelay: 10*time.Minute}
//
//	// User-facing app (balanced)
//	balanced := llm.RetryConfig{MaxRetries: 3, BaseDelay: 1*time.Second, BackoffFactor: 2.0}
