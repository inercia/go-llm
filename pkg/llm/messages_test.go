package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMessageDeepCopy(t *testing.T) {
	t.Run("text_message_deep_copy", func(t *testing.T) {
		original := NewTextMessage(RoleAssistant, "Hello World")
		original.SetMetadata("key1", "value1")
		original.SetMetadata("key2", 42)

		// Create deep copy
		copied := original.DeepCopy()

		// Verify basic properties are copied
		assert.Equal(t, original.Role, copied.Role)
		assert.Equal(t, original.GetText(), copied.GetText())
		assert.Equal(t, "Hello World", copied.GetText())

		// Verify metadata is copied
		value1, exists := copied.GetMetadata("key1")
		assert.True(t, exists)
		assert.Equal(t, "value1", value1)

		value2, exists := copied.GetMetadata("key2")
		assert.True(t, exists)
		assert.Equal(t, 42, value2)

		// Verify independence - modify original
		original.SetText("Modified Original")
		original.SetMetadata("key1", "modified")

		// Copy should be unaffected
		assert.Equal(t, "Hello World", copied.GetText())
		value1Copy, _ := copied.GetMetadata("key1")
		assert.Equal(t, "value1", value1Copy)

		t.Log("✅ Text message deep copy works correctly")
	})

	t.Run("complex_message_with_tool_calls", func(t *testing.T) {
		original := Message{
			Role: RoleAssistant,
			Content: []MessageContent{
				NewTextContent("I'll help you with that calculation."),
			},
			ToolCalls: []ToolCall{
				{
					ID:   "call_123",
					Type: "function",
					Function: ToolCallFunction{
						Name:      "calculate",
						Arguments: `{"operation": "add", "a": 5, "b": 3}`,
					},
				},
			},
			Metadata: map[string]any{
				"confidence": 0.95,
				"tokens":     150,
				"nested": map[string]any{
					"source": "assistant",
					"model":  "gpt-4",
				},
			},
		}

		// Create deep copy
		copied := original.DeepCopy()

		// Verify tool calls are copied
		require.Len(t, copied.ToolCalls, 1)
		assert.Equal(t, "call_123", copied.ToolCalls[0].ID)
		assert.Equal(t, "calculate", copied.ToolCalls[0].Function.Name)

		// Verify nested metadata is copied
		nested, exists := copied.GetMetadata("nested")
		assert.True(t, exists)
		nestedMap, ok := nested.(map[string]any)
		require.True(t, ok)
		assert.Equal(t, "assistant", nestedMap["source"])

		// Verify independence - modify original tool call
		original.ToolCalls[0].Function.Arguments = `{"operation": "multiply", "a": 10, "b": 2}`

		// Copy should be unaffected
		assert.Equal(t, `{"operation": "add", "a": 5, "b": 3}`, copied.ToolCalls[0].Function.Arguments)

		// Verify independence - modify original nested metadata
		originalNested := original.Metadata["nested"].(map[string]any)
		originalNested["source"] = "modified"

		// Copy should be unaffected
		copiedNested := copied.Metadata["nested"].(map[string]any)
		assert.Equal(t, "assistant", copiedNested["source"])

		t.Log("✅ Complex message with tool calls deep copy works correctly")
	})

	t.Run("message_with_binary_content", func(t *testing.T) {
		// Create message with image content containing binary data
		imageData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A} // PNG header
		original := Message{
			Role: RoleUser,
			Content: []MessageContent{
				NewTextContent("Please analyze this image."),
				&ImageContent{
					Data:     imageData,
					MimeType: "image/png",
				},
			},
		}

		// Create deep copy
		copied := original.DeepCopy()

		// Verify content is copied
		require.Len(t, copied.Content, 2)

		// Check text content
		textContent := copied.Content[0].(*TextContent)
		assert.Equal(t, "Please analyze this image.", textContent.GetText())

		// Check image content
		imageContent := copied.Content[1].(*ImageContent)
		assert.Equal(t, "image/png", imageContent.MimeType)
		assert.Equal(t, imageData, imageContent.Data)

		// Verify independence - modify original binary data
		original.Content[1].(*ImageContent).Data[0] = 0xFF

		// Copy should be unaffected
		copiedImageContent := copied.Content[1].(*ImageContent)
		assert.Equal(t, byte(0x89), copiedImageContent.Data[0])

		t.Log("✅ Message with binary content deep copy works correctly")
	})

	t.Run("empty_and_nil_scenarios", func(t *testing.T) {
		// Test empty message
		empty := Message{Role: RoleSystem}
		copiedEmpty := empty.DeepCopy()
		assert.Equal(t, RoleSystem, copiedEmpty.Role)
		assert.Len(t, copiedEmpty.Content, 0)
		assert.Len(t, copiedEmpty.ToolCalls, 0)
		assert.Len(t, copiedEmpty.Metadata, 0)

		// Test message with nil content
		withNilContent := Message{
			Role:    RoleUser,
			Content: nil,
		}
		copiedWithNil := withNilContent.DeepCopy()
		assert.Equal(t, RoleUser, copiedWithNil.Role)
		assert.Nil(t, copiedWithNil.Content)

		t.Log("✅ Empty and nil scenarios handled correctly")
	})

	t.Run("metadata_edge_cases", func(t *testing.T) {
		original := Message{
			Role: RoleAssistant,
			Content: []MessageContent{
				NewTextContent("Test message"),
			},
			Metadata: map[string]any{
				"string_value": "test",
				"int_value":    42,
				"float_value":  3.14,
				"bool_value":   true,
				"nil_value":    nil,
				"byte_slice":   []byte{1, 2, 3, 4},
				"string_slice": []any{"a", "b", "c"},
				"nested_map": map[string]any{
					"level2": map[string]any{
						"level3": "deep",
					},
				},
			},
		}

		copied := original.DeepCopy()

		// Verify all metadata types are copied correctly
		assert.Equal(t, "test", copied.Metadata["string_value"])
		assert.Equal(t, 42, copied.Metadata["int_value"])
		assert.Equal(t, 3.14, copied.Metadata["float_value"])
		assert.Equal(t, true, copied.Metadata["bool_value"])
		assert.Nil(t, copied.Metadata["nil_value"])

		// Verify byte slice is deep copied
		byteSlice := copied.Metadata["byte_slice"].([]byte)
		assert.Equal(t, []byte{1, 2, 3, 4}, byteSlice)

		// Modify original byte slice
		originalBytes := original.Metadata["byte_slice"].([]byte)
		originalBytes[0] = 99

		// Copy should be unaffected
		copiedBytes := copied.Metadata["byte_slice"].([]byte)
		assert.Equal(t, byte(1), copiedBytes[0])

		t.Log("✅ Metadata edge cases handled correctly")
	})
}

// TestDeepCopyWithTracing demonstrates how DeepCopy solves the tracing issue
func TestDeepCopyWithTracing(t *testing.T) {
	t.Run("tracing_with_mutable_state_protection", func(t *testing.T) {
		// Create original response message
		original := NewTextMessage(RoleAssistant, "Original response text")

		// Simulate tracing middleware that creates a copy before processing
		tracedMessage := original.DeepCopy()

		// Simulate concurrent modification of the original (as might happen in real systems)
		original.SetText("Modified after tracing")

		// Verify the traced message is unaffected
		assert.Equal(t, "Original response text", tracedMessage.GetText())
		assert.Equal(t, "Modified after tracing", original.GetText())

		t.Log("✅ DeepCopy protects against concurrent modifications in tracing")
	})

	t.Run("multiple_tracers_with_deep_copy", func(t *testing.T) {
		// Create original message
		original := NewTextMessage(RoleAssistant, "Shared message")
		original.SetMetadata("timestamp", "2024-01-01T00:00:00Z")

		// Simulate multiple tracing systems each getting their own copy
		tracer1Copy := original.DeepCopy()
		tracer2Copy := original.DeepCopy()

		// Each tracer adds its own metadata to its copy
		tracer1Copy.SetMetadata("tracer", "system1")
		tracer2Copy.SetMetadata("tracer", "system2")

		// Verify copies are independent
		tracer1Value, _ := tracer1Copy.GetMetadata("tracer")
		tracer2Value, _ := tracer2Copy.GetMetadata("tracer")

		assert.Equal(t, "system1", tracer1Value)
		assert.Equal(t, "system2", tracer2Value)

		// Original should be unaffected
		_, exists := original.GetMetadata("tracer")
		assert.False(t, exists)

		t.Log("✅ Multiple tracers can work independently with DeepCopy")
	})
}
