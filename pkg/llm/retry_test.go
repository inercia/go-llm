package llm

import (
	"context"
	"testing"
	"time"
)

// MockChatCompleter is a mock implementation for testing
type MockChatCompleter struct {
	responses        []*ChatResponse
	errors           []error
	callCount        int
	callDurations    []time.Duration
	lastCallTime     time.Time
	timeBetweenCalls []time.Duration
}

func (m *MockChatCompleter) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	now := time.Now()
	if m.callCount > 0 && !m.lastCallTime.IsZero() {
		m.timeBetweenCalls = append(m.timeBetweenCalls, now.Sub(m.lastCallTime))
	}
	m.lastCallTime = now

	// Apply call duration if specified
	if m.callCount < len(m.callDurations) {
		time.Sleep(m.callDurations[m.callCount])
	}

	if m.callCount < len(m.errors) && m.errors[m.callCount] != nil {
		err := m.errors[m.callCount]
		m.callCount++
		return nil, err
	}

	if m.callCount < len(m.responses) {
		resp := m.responses[m.callCount]
		m.callCount++
		return resp, nil
	}

	m.callCount++
	return &ChatResponse{ID: "test-response", Model: "test-model"}, nil
}

func TestRetryChatCompletion_Success(t *testing.T) {
	// Test successful call without retries
	mock := &MockChatCompleter{
		responses: []*ChatResponse{
			{ID: "success-1", Model: "test-model"},
		},
	}

	retryClient := RetryChatCompletion(mock)

	ctx := context.Background()
	req := ChatRequest{Model: "test-model"}

	resp, err := retryClient.ChatCompletion(ctx, req)

	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if resp == nil || resp.ID != "success-1" {
		t.Errorf("Expected response with ID 'success-1', got: %v", resp)
	}
	if mock.callCount != 1 {
		t.Errorf("Expected 1 call, got: %d", mock.callCount)
	}
}

func TestRetryChatCompletion_RateLimitRetry(t *testing.T) {
	// Test retry on rate limit error
	rateLimitErr := &Error{
		Code:       "rate_limit_exceeded",
		Message:    "Rate limit exceeded",
		Type:       "rate_limit_error",
		StatusCode: 429,
	}

	mock := &MockChatCompleter{
		errors: []error{
			rateLimitErr, // First call fails
			rateLimitErr, // Second call fails
			nil,          // Third call succeeds
		},
		responses: []*ChatResponse{
			nil, nil, // First two responses not used (errors)
			{ID: "retry-success", Model: "test-model"}, // Third response
		},
	}

	config := RetryConfig{
		MaxRetries:    3,
		BaseDelay:     10 * time.Millisecond, // Short delay for testing
		BackoffFactor: 2.0,
		Jitter:        false, // Disable jitter for predictable testing
	}

	retryClient := RetryChatCompletion(mock, config)

	ctx := context.Background()
	req := ChatRequest{Model: "test-model"}

	start := time.Now()
	resp, err := retryClient.ChatCompletion(ctx, req)
	duration := time.Since(start)

	if err != nil {
		t.Errorf("Expected no error after retries, got: %v", err)
	}
	if resp == nil || resp.ID != "retry-success" {
		t.Errorf("Expected response with ID 'retry-success', got: %v", resp)
	}
	if mock.callCount != 3 {
		t.Errorf("Expected 3 calls, got: %d", mock.callCount)
	}

	// Should have taken at least 10ms + 20ms = 30ms for delays
	expectedMinDuration := 30 * time.Millisecond
	if duration < expectedMinDuration {
		t.Errorf("Expected duration >= %v, got: %v", expectedMinDuration, duration)
	}
}

func TestRetryChatCompletion_ServerErrorRetry(t *testing.T) {
	// Test retry on server error (5xx)
	serverErr := &Error{
		Code:       "server_error",
		Message:    "Internal server error",
		Type:       "api_error",
		StatusCode: 502,
	}

	mock := &MockChatCompleter{
		errors: []error{
			serverErr, // First call fails with server error
			nil,       // Second call succeeds
		},
		responses: []*ChatResponse{
			nil, // First response not used (error)
			{ID: "server-retry-success", Model: "test-model"},
		},
	}

	config := RetryConfig{
		MaxRetries:    2,
		BaseDelay:     5 * time.Millisecond,
		BackoffFactor: 2.0,
		Jitter:        false,
	}

	retryClient := RetryChatCompletion(mock, config)

	ctx := context.Background()
	req := ChatRequest{Model: "test-model"}

	resp, err := retryClient.ChatCompletion(ctx, req)

	if err != nil {
		t.Errorf("Expected no error after retry, got: %v", err)
	}
	if resp == nil || resp.ID != "server-retry-success" {
		t.Errorf("Expected response with ID 'server-retry-success', got: %v", resp)
	}
	if mock.callCount != 2 {
		t.Errorf("Expected 2 calls, got: %d", mock.callCount)
	}
}

func TestRetryChatCompletion_NonRetryableError(t *testing.T) {
	// Test that non-retryable errors are not retried
	authErr := &Error{
		Code:       "invalid_api_key",
		Message:    "Invalid API key",
		Type:       "authentication_error",
		StatusCode: 401,
	}

	mock := &MockChatCompleter{
		errors: []error{authErr},
	}

	retryClient := RetryChatCompletion(mock)

	ctx := context.Background()
	req := ChatRequest{Model: "test-model"}

	_, err := retryClient.ChatCompletion(ctx, req)

	if err == nil {
		t.Error("Expected authentication error, got nil")
	}
	if mock.callCount != 1 {
		t.Errorf("Expected 1 call (no retries), got: %d", mock.callCount)
	}
}

func TestRetryChatCompletion_MaxRetriesExceeded(t *testing.T) {
	// Test that max retries limit is respected
	rateLimitErr := &Error{
		Code:       "rate_limit_exceeded",
		Message:    "Rate limit exceeded",
		Type:       "rate_limit_error",
		StatusCode: 429,
	}

	mock := &MockChatCompleter{
		errors: []error{
			rateLimitErr, rateLimitErr, rateLimitErr, rateLimitErr, // All calls fail
		},
	}

	config := RetryConfig{
		MaxRetries:    2, // Max 2 retries = 3 total attempts
		BaseDelay:     5 * time.Millisecond,
		BackoffFactor: 2.0,
		Jitter:        false,
	}

	retryClient := RetryChatCompletion(mock, config)

	ctx := context.Background()
	req := ChatRequest{Model: "test-model"}

	_, err := retryClient.ChatCompletion(ctx, req)

	if err == nil {
		t.Error("Expected error after max retries exceeded, got nil")
	}
	if mock.callCount != 3 { // 1 initial + 2 retries
		t.Errorf("Expected 3 calls (1 initial + 2 retries), got: %d", mock.callCount)
	}
}

func TestRetryChatCompletion_ContextCancellation(t *testing.T) {
	// Test that context cancellation during retry delay is respected
	rateLimitErr := &Error{
		Code:       "rate_limit_exceeded",
		Message:    "Rate limit exceeded",
		Type:       "rate_limit_error",
		StatusCode: 429,
	}

	mock := &MockChatCompleter{
		errors: []error{rateLimitErr, rateLimitErr}, // Always fail
	}

	config := RetryConfig{
		MaxRetries:    3,
		BaseDelay:     500 * time.Millisecond, // Long delay to allow cancellation
		BackoffFactor: 2.0,
		Jitter:        false,
	}

	retryClient := RetryChatCompletion(mock, config)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	req := ChatRequest{Model: "test-model"}

	start := time.Now()
	_, err := retryClient.ChatCompletion(ctx, req)
	duration := time.Since(start)

	if err == nil {
		t.Error("Expected context cancellation error, got nil")
	}
	if err != context.DeadlineExceeded {
		t.Errorf("Expected context.DeadlineExceeded, got: %v", err)
	}

	// Should have been cancelled quickly, not waited for full retry delay
	if duration > 200*time.Millisecond {
		t.Errorf("Expected quick cancellation, but took: %v", duration)
	}

	// Should have made at least one call before cancellation
	if mock.callCount < 1 {
		t.Errorf("Expected at least 1 call, got: %d", mock.callCount)
	}
}

func TestRetryChatCompletion_ExponentialBackoff(t *testing.T) {
	// Test that exponential backoff works correctly
	rateLimitErr := &Error{
		Code:       "rate_limit_exceeded",
		Message:    "Rate limit exceeded",
		Type:       "rate_limit_error",
		StatusCode: 429,
	}

	mock := &MockChatCompleter{
		errors: []error{
			rateLimitErr, // First call fails
			rateLimitErr, // Second call fails
			nil,          // Third call succeeds
		},
		responses: []*ChatResponse{
			nil, nil, // First two responses not used (errors)
			{ID: "backoff-success", Model: "test-model"},
		},
	}

	config := RetryConfig{
		MaxRetries:    3,
		BaseDelay:     20 * time.Millisecond,
		BackoffFactor: 2.0,
		Jitter:        false,
	}

	retryClient := RetryChatCompletion(mock, config)

	ctx := context.Background()
	req := ChatRequest{Model: "test-model"}

	start := time.Now()
	resp, err := retryClient.ChatCompletion(ctx, req)
	duration := time.Since(start)

	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if resp == nil || resp.ID != "backoff-success" {
		t.Errorf("Expected successful response, got: %v", resp)
	}

	// Expected delays: 20ms (first retry) + 40ms (second retry) = 60ms minimum
	expectedMinDuration := 60 * time.Millisecond
	if duration < expectedMinDuration {
		t.Errorf("Expected duration >= %v, got: %v", expectedMinDuration, duration)
	}
}

func TestRetryChatCompletion_DefaultConfig(t *testing.T) {
	// Test that default configuration works
	mock := &MockChatCompleter{
		responses: []*ChatResponse{
			{ID: "default-config", Model: "test-model"},
		},
	}

	retryClient := RetryChatCompletion(mock) // Use default config

	ctx := context.Background()
	req := ChatRequest{Model: "test-model"}

	resp, err := retryClient.ChatCompletion(ctx, req)

	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
	if resp == nil || resp.ID != "default-config" {
		t.Errorf("Expected response with ID 'default-config', got: %v", resp)
	}
}

func TestRetryChatCompletion_CustomRetryableErrors(t *testing.T) {
	// Test custom retryable error codes
	customErr := &Error{
		Code:    "custom_error",
		Message: "Custom error message",
		Type:    "custom_error_type",
	}

	mock := &MockChatCompleter{
		errors: []error{
			customErr, // First call fails with custom error
			nil,       // Second call succeeds
		},
		responses: []*ChatResponse{
			nil, // First response not used (error)
			{ID: "custom-retry-success", Model: "test-model"},
		},
	}

	config := RetryConfig{
		MaxRetries:      2,
		BaseDelay:       5 * time.Millisecond,
		BackoffFactor:   2.0,
		Jitter:          false,
		RetryableErrors: []string{"custom_error"}, // Custom retryable error
	}

	retryClient := RetryChatCompletion(mock, config)

	ctx := context.Background()
	req := ChatRequest{Model: "test-model"}

	resp, err := retryClient.ChatCompletion(ctx, req)

	if err != nil {
		t.Errorf("Expected no error after retry, got: %v", err)
	}
	if resp == nil || resp.ID != "custom-retry-success" {
		t.Errorf("Expected response with ID 'custom-retry-success', got: %v", resp)
	}
	if mock.callCount != 2 {
		t.Errorf("Expected 2 calls, got: %d", mock.callCount)
	}
}

func TestRetryableChatCompleter_ImplementsInterface(t *testing.T) {
	// Compile-time check that RetryableChatCompleter implements ChatCompleter
	var _ ChatCompleter = &RetryableChatCompleter{}

	// Also test that it works with the main Client interface
	mock := &MockChatCompleter{}
	retryClient := RetryChatCompletion(mock)

	// Should be able to use as ChatCompleter
	var chatCompleter = retryClient
	if chatCompleter == nil {
		t.Error("RetryableChatCompleter should implement ChatCompleter interface")
	}
}

func TestRetryChatCompletion_RetryOnStatusCodes(t *testing.T) {
	// Test that only specified status codes trigger retries

	// First test: Only retry on 429
	rateLimitErr := &Error{
		Code:       "rate_limit_exceeded",
		Message:    "Rate limit exceeded",
		Type:       "rate_limit_error",
		StatusCode: 429,
	}

	serverErr := &Error{
		Code:       "server_error",
		Message:    "Internal server error",
		Type:       "api_error",
		StatusCode: 500,
	}

	mock := &MockChatCompleter{
		errors: []error{
			rateLimitErr, // First call fails with 429 - should be retried
			nil,          // Second call succeeds
		},
		responses: []*ChatResponse{
			nil, // First response not used (error)
			{ID: "success-after-429", Model: "test-model"},
		},
	}

	config := RetryConfig{
		MaxRetries:         2,
		BaseDelay:          5 * time.Millisecond,
		Jitter:             false,
		RetryOnStatusCodes: []int{429}, // Only retry on 429
	}

	retryClient := RetryChatCompletion(mock, config)

	ctx := context.Background()
	req := ChatRequest{Model: "test-model"}

	// Should retry and succeed on 429 error
	resp, err := retryClient.ChatCompletion(ctx, req)
	if err != nil {
		t.Errorf("Expected success after retrying 429, got: %v", err)
	}
	if resp == nil || resp.ID != "success-after-429" {
		t.Errorf("Expected successful response, got: %v", resp)
	}
	if mock.callCount != 2 {
		t.Errorf("Expected 2 calls (429 retry), got: %d", mock.callCount)
	}

	// Reset mock for second test: Should NOT retry on 500 error
	mock = &MockChatCompleter{
		errors: []error{serverErr}, // 500 error - should NOT be retried
	}
	retryClient = RetryChatCompletion(mock, config)

	_, err = retryClient.ChatCompletion(ctx, req)
	if err == nil {
		t.Error("Expected 500 error to not be retried, but it succeeded")
	}
	if mock.callCount != 1 {
		t.Errorf("Expected 1 call (no retry on 500), got: %d", mock.callCount)
	}
}

func TestRetryChatCompletion_RetryOnErrorTypes(t *testing.T) {
	// Test that only specified error types trigger retries

	rateLimitErr := &Error{
		Code:       "rate_limit_exceeded",
		Message:    "Rate limit exceeded",
		Type:       "rate_limit_error",
		StatusCode: 429,
	}

	apiErr := &Error{
		Code:       "api_error",
		Message:    "API error",
		Type:       "api_error",
		StatusCode: 500,
	}

	authErr := &Error{
		Code:       "invalid_api_key",
		Message:    "Invalid API key",
		Type:       "authentication_error",
		StatusCode: 401,
	}

	mock := &MockChatCompleter{
		errors: []error{
			rateLimitErr, // First call fails with rate_limit_error - should be retried
			nil,          // Second call succeeds
		},
		responses: []*ChatResponse{
			nil, // First response not used (error)
			{ID: "success-after-rate-limit", Model: "test-model"},
		},
	}

	config := RetryConfig{
		MaxRetries:        2,
		BaseDelay:         5 * time.Millisecond,
		Jitter:            false,
		RetryOnErrorTypes: []string{"rate_limit_error"}, // Only retry on rate_limit_error
	}

	retryClient := RetryChatCompletion(mock, config)

	ctx := context.Background()
	req := ChatRequest{Model: "test-model"}

	// Should retry and succeed on rate_limit_error type
	resp, err := retryClient.ChatCompletion(ctx, req)
	if err != nil {
		t.Errorf("Expected success after retrying rate_limit_error, got: %v", err)
	}
	if resp == nil || resp.ID != "success-after-rate-limit" {
		t.Errorf("Expected successful response, got: %v", resp)
	}
	if mock.callCount != 2 {
		t.Errorf("Expected 2 calls (rate_limit_error retry), got: %d", mock.callCount)
	}

	// Test that api_error is NOT retried
	mock = &MockChatCompleter{
		errors: []error{apiErr}, // api_error - should NOT be retried
	}
	retryClient = RetryChatCompletion(mock, config)

	_, err = retryClient.ChatCompletion(ctx, req)
	if err == nil {
		t.Error("Expected api_error to not be retried, but it succeeded")
	}
	if mock.callCount != 1 {
		t.Errorf("Expected 1 call (no retry on api_error), got: %d", mock.callCount)
	}

	// Test that auth_error is NOT retried
	mock = &MockChatCompleter{
		errors: []error{authErr}, // auth_error - should NOT be retried
	}
	retryClient = RetryChatCompletion(mock, config)

	_, err = retryClient.ChatCompletion(ctx, req)
	if err == nil {
		t.Error("Expected authentication_error to not be retried, but it succeeded")
	}
	if mock.callCount != 1 {
		t.Errorf("Expected 1 call (no retry on authentication_error), got: %d", mock.callCount)
	}
}

func TestRetryChatCompletion_CombinedStatusCodesAndErrorTypes(t *testing.T) {
	// Test that both status codes AND error types can be used together

	serverErrWithAPIType := &Error{
		Code:       "server_error",
		Message:    "Server error",
		Type:       "api_error", // This type should trigger retry
		StatusCode: 500,         // This status should NOT trigger retry (not in list)
	}

	serverErrWith502 := &Error{
		Code:       "server_error",
		Message:    "Bad gateway",
		Type:       "some_other_type", // This type should NOT trigger retry
		StatusCode: 502,               // This status should trigger retry
	}

	mock := &MockChatCompleter{
		errors: []error{
			serverErrWithAPIType, // Should be retried due to "api_error" type
			nil,                  // Second call succeeds
		},
		responses: []*ChatResponse{
			nil, // First response not used (error)
			{ID: "success-api-error-type", Model: "test-model"},
		},
	}

	config := RetryConfig{
		MaxRetries:         2,
		BaseDelay:          5 * time.Millisecond,
		Jitter:             false,
		RetryOnStatusCodes: []int{429, 502},       // Retry on 429 and 502
		RetryOnErrorTypes:  []string{"api_error"}, // Retry on api_error type
	}

	retryClient := RetryChatCompletion(mock, config)

	ctx := context.Background()
	req := ChatRequest{Model: "test-model"}

	// Should retry due to api_error type (even though 500 is not in status codes list)
	resp, err := retryClient.ChatCompletion(ctx, req)
	if err != nil {
		t.Errorf("Expected success after retrying api_error type, got: %v", err)
	}
	if resp == nil || resp.ID != "success-api-error-type" {
		t.Errorf("Expected successful response, got: %v", resp)
	}
	if mock.callCount != 2 {
		t.Errorf("Expected 2 calls (api_error retry), got: %d", mock.callCount)
	}

	// Test that 502 status code is retried (even though type is not in list)
	mock = &MockChatCompleter{
		errors: []error{
			serverErrWith502, // Should be retried due to 502 status code
			nil,              // Second call succeeds
		},
		responses: []*ChatResponse{
			nil, // First response not used (error)
			{ID: "success-502-status", Model: "test-model"},
		},
	}

	retryClient = RetryChatCompletion(mock, config)

	resp, err = retryClient.ChatCompletion(ctx, req)
	if err != nil {
		t.Errorf("Expected success after retrying 502 status, got: %v", err)
	}
	if resp == nil || resp.ID != "success-502-status" {
		t.Errorf("Expected successful response, got: %v", resp)
	}
	if mock.callCount != 2 {
		t.Errorf("Expected 2 calls (502 status retry), got: %d", mock.callCount)
	}
}

func TestRetryChatCompletion_BackwardCompatibility(t *testing.T) {
	// Test that the old behavior still works when new fields are not set

	rateLimitErr := &Error{
		Code:       "rate_limit_exceeded",
		Message:    "Rate limit exceeded",
		Type:       "rate_limit_error",
		StatusCode: 429,
	}

	serverErr := &Error{
		Code:       "server_error",
		Message:    "Internal server error",
		Type:       "api_error",
		StatusCode: 500,
	}

	// Test that 429 is retried (backward compatibility)
	mock := &MockChatCompleter{
		errors:    []error{rateLimitErr, nil},
		responses: []*ChatResponse{nil, {ID: "success-429", Model: "test-model"}},
	}

	config := RetryConfig{
		MaxRetries: 2,
		BaseDelay:  5 * time.Millisecond,
		Jitter:     false,
		// No RetryOnStatusCodes or RetryOnErrorTypes - should use old behavior
	}

	retryClient := RetryChatCompletion(mock, config)

	ctx := context.Background()
	req := ChatRequest{Model: "test-model"}

	_, err := retryClient.ChatCompletion(ctx, req)
	if err != nil {
		t.Errorf("Expected backward compatibility success on 429, got: %v", err)
	}
	if mock.callCount != 2 {
		t.Errorf("Expected 2 calls (backward compatibility), got: %d", mock.callCount)
	}

	// Test that 5xx is retried (backward compatibility)
	mock = &MockChatCompleter{
		errors:    []error{serverErr, nil},
		responses: []*ChatResponse{nil, {ID: "success-500", Model: "test-model"}},
	}

	retryClient = RetryChatCompletion(mock, config)

	_, err = retryClient.ChatCompletion(ctx, req)
	if err != nil {
		t.Errorf("Expected backward compatibility success on 500, got: %v", err)
	}
	if mock.callCount != 2 {
		t.Errorf("Expected 2 calls (backward compatibility 5xx), got: %d", mock.callCount)
	}
}
