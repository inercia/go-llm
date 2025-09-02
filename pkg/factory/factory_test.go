package factory

import (
	"testing"

	"github.com/inercia/go-llm/pkg/llm"
)

// TestFactory tests the factory functionality
func TestFactory(t *testing.T) {
	t.Parallel()

	t.Run("CreateClient_Validation", func(t *testing.T) {
		t.Parallel()

		factory := New()

		// Test missing model - should return validation error
		_, err := factory.CreateClient(llm.ClientConfig{Provider: "nonexistent"})
		if err == nil {
			t.Error("Expected error for missing model")
		}

		// Verify it's a validation error
		if llmErr, ok := err.(*llm.Error); ok {
			if llmErr.Type != "validation_error" {
				t.Errorf("Expected validation_error, got %s", llmErr.Type)
			}
		} else {
			t.Errorf("Expected *llm.Error type, got %T", err)
		}
	})

	t.Run("Factory Basic Functionality", func(t *testing.T) {
		t.Parallel()
		factory := New()

		// Test that factory exists and can be created
		if factory == nil {
			t.Error("Expected factory to be created")
		}

		// Test unsupported provider error
		_, err := factory.CreateClient(llm.ClientConfig{
			Provider: "unsupported",
			Model:    "some-model",
		})
		if err == nil {
			t.Error("Expected error for unsupported provider")
		}

		// Verify error type
		if llmErr, ok := err.(*llm.Error); ok {
			if llmErr.Code != "unsupported_provider" && llmErr.Code != "missing_model" {
				t.Errorf("Expected unsupported_provider or missing_model error, got %s", llmErr.Code)
			}
		}
	})

	t.Run("Auto Registration Works", func(t *testing.T) {
		t.Parallel()

		// Since the registry package imports all providers via imports.go,
		// they should all be automatically registered
		providers := ListProviders()

		if len(providers) == 0 {
			t.Error("Expected providers to be auto-registered, but none found")
		}

		// Test that we can create a mock client (should always be available)
		factory := New()
		_, err := factory.CreateClient(llm.ClientConfig{
			Provider: "mock",
			Model:    "test-model",
		})
		if err != nil {
			t.Errorf("Failed to create mock client with auto-registered provider: %v", err)
		}
	})

	// Note: Actual provider integration tests are in separate integration test files
	// This test validates the factory logic and auto-registration functionality.
}
