package llm

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// ExtractJSONFromResponse extracts JSON from LLM response that may contain markdown
// code blocks or other text. It returns the extracted JSON string or the original
// response if no JSON is found.
//
// Example:
//
//	response := "Here is the data:\n```json\n{\"key\": \"value\"}\n```"
//	jsonStr := ExtractJSONFromResponse(response)
//	fmt.Println(jsonStr) // Output: {"key": "value"}
func ExtractJSONFromResponse(response string) string {
	// Remove color codes and ANSI escape sequences
	response = strings.ReplaceAll(response, "\033[92m", "")
	response = strings.ReplaceAll(response, "\033[0m", "")
	response = strings.ReplaceAll(response, "[92m", "")
	response = strings.ReplaceAll(response, "[0m", "")

	// Try to find JSON in code blocks first
	jsonBlockRegex := regexp.MustCompile("```json\\s*([\\s\\S]*?)```")
	matches := jsonBlockRegex.FindStringSubmatch(response)
	if len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}

	// Try to find JSON-like structure in the response
	// Find from first { to last } in the response
	start := strings.Index(response, "{")
	if start == -1 {
		return response // Return original if no JSON found
	}

	// Find the last closing brace
	end := strings.LastIndex(response, "}")
	if end == -1 || end <= start {
		return response // Return original if no complete JSON found
	}

	jsonContent := response[start : end+1]
	return jsonContent
}

// ExtractJSONToStruct extracts JSON from LLM response and unmarshals it into the provided struct.
// The out parameter should be a pointer to the struct.
//
// Example:
//
//	var result MyStruct
//	err := ExtractJSONToStruct(response, &result)
//	if err != nil {
//	    // handle error
//	}
func ExtractJSONToStruct(response string, out interface{}) error {
	jsonStr := ExtractJSONFromResponse(response)
	return json.Unmarshal([]byte(jsonStr), out)
}

// RemoveBlocks removes all blocks of the specified tag from the input string.
// For example, RemoveBlocks(text, "think") will remove all <think>...</think> blocks.
//
// Example:
//
//	text := "Hello <think>this is internal</think> world!"
//	cleaned := RemoveBlocks(text, "think")
//	fmt.Println(cleaned) // Output: "Hello  world!"
func RemoveBlocks(text, tag string) string {
	// Create regex pattern to match <tag>...</tag> blocks
	// Using (?s) to make . match newlines as well
	pattern := fmt.Sprintf(`(?s)<%s>.*?</%s>`, regexp.QuoteMeta(tag), regexp.QuoteMeta(tag))
	regex := regexp.MustCompile(pattern)
	return regex.ReplaceAllString(text, "")
}
