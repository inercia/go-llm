# Streaming Chat Example

This example demonstrates how to use streaming chat completions with the go-llm library. It shows the difference between streaming and non-streaming responses, providing real-time output as the LLM generates text.

## What it does

1. **Auto-detects provider** using environment variables and creates a client
2. **Checks streaming support** for the selected model
3. **Makes a streaming request** asking for a short story about a robot learning to dance
4. **Processes stream events** in real-time, printing text chunks as they arrive
5. **Compares with non-streaming** by making a follow-up question without streaming

## Key Features Demonstrated

- **Real-time streaming**: Text appears character by character as the model generates it
- **Stream event handling**: Shows how to handle different event types (`IsDelta()`, `IsDone()`, `IsError()`)
- **Content accumulation**: Builds the full response while streaming individual chunks
- **Error handling**: Proper error handling for both streaming and non-streaming requests
- **Resource cleanup**: Shows proper client cleanup with defer statements

## Running the Example

1. Set your API key for any supported provider (the example auto-detects):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"  # or GEMINI_API_KEY, DEEPSEEK_API_KEY, etc.
   ```
2. Run the example:
   ```bash
   go run main.go
   ```

**Note**: The example automatically detects which LLM provider to use based on available environment variables. It works with any provider that supports streaming.

## Expected Output

The example will:

- Stream a creative story about a robot learning to dance
- Show each text chunk appearing in real-time
- Display completion statistics (finish reason, full story)
- Follow up with a non-streaming question about the story's moral
- Show token usage information

This demonstrates the library's ability to provide both streaming and traditional chat completion modes with a unified interface.
