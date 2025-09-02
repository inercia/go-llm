package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatResponseDeepCopy(t *testing.T) {
	t.Run("basic_response_deep_copy", func(t *testing.T) {
		original := &ChatResponse{
			ID:    "chatcmpl-test-123",
			Model: "openai/gpt-oss-20b",
			Choices: []Choice{
				{
					Index:        0,
					Message:      NewTextMessage(RoleAssistant, "Hello, how can I help you today?"),
					FinishReason: "stop",
				},
			},
			Usage: Usage{
				PromptTokens:     25,
				CompletionTokens: 12,
				TotalTokens:      37,
			},
		}

		// Create deep copy
		copied := original.DeepCopy()

		// Verify basic properties are copied
		assert.Equal(t, original.ID, copied.ID)
		assert.Equal(t, original.Model, copied.Model)
		assert.Equal(t, original.Usage.TotalTokens, copied.Usage.TotalTokens)
		assert.Len(t, copied.Choices, 1)
		assert.Equal(t, "Hello, how can I help you today?", copied.Choices[0].Message.GetText())

		// Verify independence - modify original
		original.Choices[0].Message.SetText("Modified original response")
		original.Usage.TotalTokens = 999

		// Copy should be unaffected
		assert.Equal(t, "Hello, how can I help you today?", copied.Choices[0].Message.GetText())
		assert.Equal(t, 37, copied.Usage.TotalTokens)

		t.Log("✅ Basic ChatResponse deep copy works correctly")
	})

	t.Run("multi_choice_response_deep_copy", func(t *testing.T) {
		original := &ChatResponse{
			ID:    "chatcmpl-multi-456",
			Model: "openai/gpt-4",
			Choices: []Choice{
				{
					Index:        0,
					Message:      NewTextMessage(RoleAssistant, "First choice response"),
					FinishReason: "stop",
				},
				{
					Index:        1,
					Message:      NewTextMessage(RoleAssistant, "Second choice response"),
					FinishReason: "length",
				},
			},
			Usage: Usage{
				PromptTokens:     50,
				CompletionTokens: 30,
				TotalTokens:      80,
			},
		}

		// Create deep copy
		copied := original.DeepCopy()

		// Verify all choices are copied
		require.Len(t, copied.Choices, 2)
		assert.Equal(t, "First choice response", copied.Choices[0].Message.GetText())
		assert.Equal(t, "Second choice response", copied.Choices[1].Message.GetText())
		assert.Equal(t, "stop", copied.Choices[0].FinishReason)
		assert.Equal(t, "length", copied.Choices[1].FinishReason)

		// Verify independence - modify original first choice
		original.Choices[0].Message.SetText("Modified first choice")
		original.Choices[1].FinishReason = "modified"

		// Copy should be unaffected
		assert.Equal(t, "First choice response", copied.Choices[0].Message.GetText())
		assert.Equal(t, "length", copied.Choices[1].FinishReason)

		t.Log("✅ Multi-choice ChatResponse deep copy works correctly")
	})

	t.Run("response_with_tool_calls_deep_copy", func(t *testing.T) {
		assistantMessage := NewTextMessage(RoleAssistant, "I'll calculate that for you.")
		assistantMessage.AddToolCall(ToolCall{
			ID:   "call_abc123",
			Type: "function",
			Function: ToolCallFunction{
				Name:      "calculator",
				Arguments: `{"operation": "add", "a": 5, "b": 3}`,
			},
		})

		original := &ChatResponse{
			ID:    "chatcmpl-tools-789",
			Model: "openai/gpt-4-tools",
			Choices: []Choice{
				{
					Index:        0,
					Message:      assistantMessage,
					FinishReason: "tool_calls",
				},
			},
			Usage: Usage{
				PromptTokens:     40,
				CompletionTokens: 15,
				TotalTokens:      55,
			},
		}

		// Create deep copy
		copied := original.DeepCopy()

		// Verify tool calls are copied
		require.Len(t, copied.Choices, 1)
		require.Len(t, copied.Choices[0].Message.ToolCalls, 1)

		toolCall := copied.Choices[0].Message.ToolCalls[0]
		assert.Equal(t, "call_abc123", toolCall.ID)
		assert.Equal(t, "calculator", toolCall.Function.Name)
		assert.Equal(t, `{"operation": "add", "a": 5, "b": 3}`, toolCall.Function.Arguments)

		// Verify independence - modify original tool call
		original.Choices[0].Message.ToolCalls[0].Function.Arguments = `{"operation": "multiply", "a": 10, "b": 2}`

		// Copy should be unaffected
		copiedToolCall := copied.Choices[0].Message.ToolCalls[0]
		assert.Equal(t, `{"operation": "add", "a": 5, "b": 3}`, copiedToolCall.Function.Arguments)

		t.Log("✅ ChatResponse with tool calls deep copy works correctly")
	})

	t.Run("empty_response_deep_copy", func(t *testing.T) {
		original := &ChatResponse{
			ID:      "chatcmpl-empty-000",
			Model:   "test-model",
			Choices: []Choice{},
			Usage: Usage{
				PromptTokens:     0,
				CompletionTokens: 0,
				TotalTokens:      0,
			},
		}

		// Create deep copy
		copied := original.DeepCopy()

		// Verify empty response is handled correctly
		assert.Equal(t, original.ID, copied.ID)
		assert.Equal(t, original.Model, copied.Model)
		assert.Len(t, copied.Choices, 0)
		assert.Equal(t, 0, copied.Usage.TotalTokens)

		t.Log("✅ Empty ChatResponse deep copy works correctly")
	})
}

// TestChatResponseDeepCopyWithTracing demonstrates how DeepCopy solves the tracing issue for responses
func TestChatResponseDeepCopyWithTracing(t *testing.T) {
	t.Run("tracing_response_protection", func(t *testing.T) {
		// Create original response
		original := &ChatResponse{
			ID:    "chatcmpl-trace-test",
			Model: "openai/gpt-oss-20b",
			Choices: []Choice{
				{
					Index:        0,
					Message:      NewTextMessage(RoleAssistant, "Original traced response"),
					FinishReason: "stop",
				},
			},
			Usage: Usage{
				PromptTokens:     20,
				CompletionTokens: 10,
				TotalTokens:      30,
			},
		}

		// Simulate tracing middleware that creates a copy before processing
		tracedResponse := original.DeepCopy()

		// Simulate concurrent modification of the original (as might happen in real systems)
		original.Choices[0].Message.SetText("Modified after tracing")
		original.Usage.TotalTokens = 999
		original.ID = "modified-id"

		// Verify the traced response is unaffected
		assert.Equal(t, "Original traced response", tracedResponse.Choices[0].Message.GetText())
		assert.Equal(t, 30, tracedResponse.Usage.TotalTokens)
		assert.Equal(t, "chatcmpl-trace-test", tracedResponse.ID)

		// Verify original was indeed modified
		assert.Equal(t, "Modified after tracing", original.Choices[0].Message.GetText())
		assert.Equal(t, 999, original.Usage.TotalTokens)
		assert.Equal(t, "modified-id", original.ID)

		t.Log("✅ ChatResponse DeepCopy protects against concurrent modifications in tracing")
	})

	t.Run("simulate_real_tracing_scenario", func(t *testing.T) {
		// Create a response that simulates a real LLM response
		original := &ChatResponse{
			ID:    "chatcmpl-14rsdzsqecyiztgjdni6tp", // Same ID from the original issue
			Model: "openai/gpt-oss-20b",
			Choices: []Choice{
				{
					Index:        0,
					Message:      NewTextMessage(RoleAssistant, "Hello World"),
					FinishReason: "stop",
				},
			},
			Usage: Usage{
				PromptTokens:     76,
				CompletionTokens: 17,
				TotalTokens:      93,
			},
		}

		// Simulate the tracing process described in the original issue
		tracingCopy := original.DeepCopy()

		// Extract content for tracing (simulates extractLLMResponseContent function)
		content := map[string]interface{}{
			"response_id":       tracingCopy.ID,
			"model":             tracingCopy.Model,
			"choice_count":      len(tracingCopy.Choices),
			"total_tokens":      tracingCopy.Usage.TotalTokens,
			"completion_tokens": tracingCopy.Usage.CompletionTokens,
		}

		if len(tracingCopy.Choices) > 0 {
			choices := make([]map[string]interface{}, 0, len(tracingCopy.Choices))
			for _, choice := range tracingCopy.Choices {
				choiceContent := map[string]interface{}{
					"index":         choice.Index,
					"finish_reason": choice.FinishReason,
					"content_size":  0,
					"text_preview":  "",
				}

				// Extract text content
				text := choice.Message.GetText()
				if text != "" {
					choiceContent["text_preview"] = text
					choiceContent["content_size"] = len(text)
				}

				choices = append(choices, choiceContent)
			}
			content["choices"] = choices
		}

		// Verify the traced content has the expected structure
		assert.Equal(t, "chatcmpl-14rsdzsqecyiztgjdni6tp", content["response_id"])
		assert.Equal(t, 1, content["choice_count"])
		assert.Equal(t, 93, content["total_tokens"])

		choices, ok := content["choices"].([]map[string]interface{})
		require.True(t, ok)
		require.Len(t, choices, 1)

		choice := choices[0]
		assert.Equal(t, "Hello World", choice["text_preview"])
		assert.Equal(t, 11, choice["content_size"]) // len("Hello World")
		assert.Equal(t, "stop", choice["finish_reason"])

		t.Logf("✅ Traced content: %+v", content)
		t.Log("✅ Real tracing scenario with DeepCopy produces correct non-empty content")
	})
}
