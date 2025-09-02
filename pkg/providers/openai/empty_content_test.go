package openai

import (
	"testing"

	"github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inercia/go-llm/pkg/llm"
)

// TestConvertMessagesEmptyContent tests that empty content is handled properly
func TestConvertMessagesEmptyContent(t *testing.T) {
	client := &Client{
		model:    "gpt-3.5-turbo",
		provider: "openai",
		baseURL:  "https://api.openai.com/v1",
	}

	t.Run("single_empty_text_content", func(t *testing.T) {
		// Create a message with empty text content
		messages := []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.MessageContent{
					llm.NewTextContent(""), // Empty text
				},
			},
		}

		// Convert to OpenAI format
		openaiMessages := client.convertMessages(messages)

		require.Len(t, openaiMessages, 1, "Should have one message")

		// The empty text should be converted to a space to avoid "undefined" API error
		assert.Equal(t, " ", openaiMessages[0].Content, "Empty content should be converted to space")
		assert.Nil(t, openaiMessages[0].MultiContent, "MultiContent should be nil for simple content")
	})

	t.Run("single_whitespace_only_text_content", func(t *testing.T) {
		// Create a message with whitespace-only text content
		messages := []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.MessageContent{
					llm.NewTextContent("   \t\n   "), // Whitespace only
				},
			},
		}

		// Convert to OpenAI format
		openaiMessages := client.convertMessages(messages)

		require.Len(t, openaiMessages, 1, "Should have one message")

		// The whitespace-only text should be converted to a space to avoid "undefined" API error
		assert.Equal(t, " ", openaiMessages[0].Content, "Whitespace-only content should be converted to space")
		assert.Nil(t, openaiMessages[0].MultiContent, "MultiContent should be nil for simple content")
	})

	t.Run("multimodal_all_empty_text", func(t *testing.T) {
		// Create a message with multiple empty text contents
		messages := []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.MessageContent{
					llm.NewTextContent(""),    // Empty
					llm.NewTextContent("   "), // Whitespace only
				},
			},
		}

		// Convert to OpenAI format
		openaiMessages := client.convertMessages(messages)

		require.Len(t, openaiMessages, 1, "Should have one message")

		// All content was empty, so should fallback to space
		assert.Equal(t, " ", openaiMessages[0].Content, "All empty content should fallback to space")
		assert.Nil(t, openaiMessages[0].MultiContent, "MultiContent should be nil when all parts are empty")
	})

	t.Run("multimodal_mixed_content", func(t *testing.T) {
		// Create a message with mixed empty and non-empty text contents
		messages := []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.MessageContent{
					llm.NewTextContent(""),      // Empty - should be filtered
					llm.NewTextContent("Hello"), // Valid text
					llm.NewTextContent("   "),   // Whitespace only - should be filtered
					llm.NewTextContent("World"), // Valid text
				},
			},
		}

		// Convert to OpenAI format
		openaiMessages := client.convertMessages(messages)

		require.Len(t, openaiMessages, 1, "Should have one message")

		// Should use MultiContent with only non-empty parts
		assert.Empty(t, openaiMessages[0].Content, "Content should be empty when using MultiContent")
		require.NotNil(t, openaiMessages[0].MultiContent, "MultiContent should be set")
		require.Len(t, openaiMessages[0].MultiContent, 2, "Should have 2 non-empty text parts")

		// Check the text parts
		assert.Equal(t, openai.ChatMessagePartTypeText, openaiMessages[0].MultiContent[0].Type)
		assert.Equal(t, "Hello", openaiMessages[0].MultiContent[0].Text)
		assert.Equal(t, openai.ChatMessagePartTypeText, openaiMessages[0].MultiContent[1].Type)
		assert.Equal(t, "World", openaiMessages[0].MultiContent[1].Text)
	})

	t.Run("valid_single_text_content", func(t *testing.T) {
		// Ensure valid content still works correctly
		messages := []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.MessageContent{
					llm.NewTextContent("Hello, world!"),
				},
			},
		}

		// Convert to OpenAI format
		openaiMessages := client.convertMessages(messages)

		require.Len(t, openaiMessages, 1, "Should have one message")

		// Valid content should pass through unchanged
		assert.Equal(t, "Hello, world!", openaiMessages[0].Content, "Valid content should pass through")
		assert.Nil(t, openaiMessages[0].MultiContent, "MultiContent should be nil for simple content")
	})
}
