package llm

import (
	"context"
	"testing"
)

func TestMockClientIntegration(t *testing.T) {
	t.Parallel()

	t.Run("MockClientWithChatCompletion", func(t *testing.T) {
		t.Parallel()
		mockClient := NewMockClient("test-model", "mock")
		mockClient.WithSimpleResponse("Hello from mock!")

		req := ChatRequest{
			Model: "test-model",
			Messages: []Message{
				{Role: RoleUser, Content: []MessageContent{NewTextContent("Test message")}},
			},
		}

		resp, err := mockClient.ChatCompletion(context.Background(), req)
		if err != nil {
			t.Errorf("Mock chat completion failed: %v", err)
		}

		if resp == nil || len(resp.Choices) == 0 {
			t.Error("Expected response with choices")
		} else {
			if resp.Choices[0].Message.GetText() != "Hello from mock!" {
				t.Errorf("Expected 'Hello from mock!', got '%s'", resp.Choices[0].Message.Content)
			}
		}

		// Verify call logging
		calls := mockClient.GetCallLog()
		if len(calls) != 1 {
			t.Errorf("Expected 1 call logged, got %d", len(calls))
		}

		_ = mockClient.Close()
	})
}
