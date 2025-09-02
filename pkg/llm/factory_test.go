package llm

import (
	"testing"
)

// TestFactory tests the factory functionality
func TestFactory(t *testing.T) {
	t.Parallel()

	t.Run("CreateClient_Validation", func(t *testing.T) {
		t.Parallel()

		factory := NewFactory()

		// Test missing model - should return validation error
		_, err := factory.CreateClient(ClientConfig{Provider: "nonexistent"})
		if err == nil {
			t.Error("Expected error for missing model")
		}

		// Verify it's a validation error
		if llmErr, ok := err.(*Error); ok {
			if llmErr.Type != "validation_error" {
				t.Errorf("Expected validation_error, got %s", llmErr.Type)
			}
		} else {
			t.Errorf("Expected *Error type, got %T", err)
		}
	})

	t.Run("Factory Basic Functionality", func(t *testing.T) {
		t.Parallel()
		factory := NewFactory()

		// Test that factory exists and can be created
		if factory == nil {
			t.Error("Expected factory to be created")
		}

		// Test unsupported provider error
		_, err := factory.CreateClient(ClientConfig{
			Provider: "unsupported",
			Model:    "some-model",
		})
		if err == nil {
			t.Error("Expected error for unsupported provider")
		}

		// Verify error type
		if llmErr, ok := err.(*Error); ok {
			if llmErr.Code != "unsupported_provider" && llmErr.Code != "missing_model" {
				t.Errorf("Expected unsupported_provider or missing_model error, got %s", llmErr.Code)
			}
		}
	})

	// Note: Actual provider integration tests are in separate integration test files
	// to avoid import cycle issues. This test only validates the factory logic.
}
