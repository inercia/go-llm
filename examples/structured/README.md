# Structured Output Example

This example demonstrates how to use structured outputs with JSON Schema to get reliable, typed responses from LLMs. It shows various use cases where you need the AI to respond in a specific format that can be programmatically processed.

## What it does

The example includes **4 different demonstrations**:

### 1. Text Analysis with JSON Schema

- Analyzes product reviews with sentiment, keywords, and metadata
- Uses strict JSON Schema validation with struct tags
- Returns structured `AnalysisResult` with confidence scores, language detection, etc.

### 2. Basic JSON Mode

- Simple JSON generation without strict schema validation
- Shows how to use basic JSON mode for less structured requirements
- Demonstrates JSON extraction and pretty-printing

## Key Features Demonstrated

- **JSON Schema Generation**: Automatically creates schemas from Go structs using `swaggest/jsonschema-go`
- **Struct Validation**: Shows both strict validation (`NewJSONSchemaResponseFormatStrictFromStruct`) and basic extraction (`ExtractJSONToStruct`)
- **Multi-provider Support**: Works with any provider (OpenAI, Gemini, Ollama, etc.) - just change the `Provider` field
- **Error Handling**: Proper validation and error handling for malformed responses
- **Real-world Use Cases**: Practical examples you can adapt for your applications

## Running the Example

1. Set your API key for any supported provider (the example auto-detects):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"  # or GEMINI_API_KEY, DEEPSEEK_API_KEY, etc.
   ```
2. Run the example:
   ```bash
   go run main.go
   ```

**Note**: The example automatically detects which LLM provider to use based on available environment variables and selects the best model for structured outputs.

## Struct Tags and Schema Features

The example shows how to use struct tags for schema definition:

```go
type AnalysisResult struct {
    Sentiment  string   `json:"sentiment" description:"The overall sentiment"`
    Confidence float64  `json:"confidence" minimum:"0" maximum:"1"`
    Keywords   []string `json:"keywords" description:"Important keywords"`
    Summary    string   `json:"summary" maxLength:"200"`
}
```

Supported tags:

- `description`: Field description for the AI
- `minimum`/`maximum`: Numeric constraints
- `maxLength`/`minLength`: String length constraints
- `enum`: Allowed values for fields
- `minItems`/`maxItems`: Array size constraints

## Expected Output

Each example will show:

- The structured request being made
- The AI's response parsed into the appropriate Go struct
- Validation confirmation that the response matches the expected schema
- Pretty-printed results showing all extracted fields

This demonstrates how to reliably extract structured data from LLM responses for use in production applications.
