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
func ExtractJSONFromResponse(text string) string {
	// Remove color codes and ANSI escape sequences
	text = strings.ReplaceAll(text, "\033[92m", "")
	text = strings.ReplaceAll(text, "\033[0m", "")
	text = strings.ReplaceAll(text, "[92m", "")
	text = strings.ReplaceAll(text, "[0m", "")

	// Remove leading/trailing whitespace
	text = strings.TrimSpace(text)

	// Try to find JSON within markdown code blocks (more flexible patterns)
	patterns := []string{
		// Markdown code blocks with json label (with optional newlines)
		"```json\\s*([\\s\\S]*?)```",
		"```JSON\\s*([\\s\\S]*?)```",
		// Code blocks with other common language identifiers that might contain JSON
		"```javascript\\s*([\\s\\S]*?)```",
		"```js\\s*([\\s\\S]*?)```",
		// Generic code blocks (any language or no language)
		"```\\w*\\s*([\\s\\S]*?)```",
		// Single backticks
		"`([^`]+)`",
	}

	for _, pattern := range patterns {
		re, err := regexp.Compile(pattern)
		if err != nil {
			continue
		}

		matches := re.FindStringSubmatch(text)
		if len(matches) > 1 {
			candidate := strings.TrimSpace(matches[1])
			if isValidJSONStart(candidate) {
				// Try to preserve original formatting if JSON is already valid
				if isValidJSON(candidate) {
					return candidate
				}
				// Clean up the JSON by removing comments and extra formatting
				cleaned := cleanJSON(candidate)
				if cleaned != "" {
					return cleaned
				}
			}
		}
	}

	// Try to find complete JSON objects/arrays using a more robust approach
	jsonCandidates := findJSONBlocks(text)
	for _, candidate := range jsonCandidates {
		candidate = strings.TrimSpace(candidate)
		if isValidJSONStart(candidate) {
			if isValidJSON(candidate) {
				return candidate
			}
			cleaned := cleanJSON(candidate)
			if cleaned != "" {
				return cleaned
			}
		}
	}

	// If no JSON patterns found, check if the entire text is JSON
	if isValidJSONStart(text) {
		if isValidJSON(text) {
			return text
		}
		cleaned := cleanJSON(text)
		if cleaned != "" {
			return cleaned
		}
	}

	// Return original text if no JSON extraction possible
	return text
}

// isValidJSONStart checks if text starts with valid JSON characters
func isValidJSONStart(text string) bool {
	text = strings.TrimSpace(text)
	return strings.HasPrefix(text, "{") || strings.HasPrefix(text, "[")
}

// isValidJSON checks if the text is valid JSON by attempting to parse it
func isValidJSON(text string) bool {
	var temp interface{}
	return json.Unmarshal([]byte(text), &temp) == nil
}

// findJSONBlocks uses a more robust approach to find JSON objects and arrays
func findJSONBlocks(text string) []string {
	var results []string

	// Find JSON objects with proper bracket matching
	for i := 0; i < len(text); i++ {
		if text[i] == '{' {
			// Find matching closing brace
			braceCount := 1
			inString := false
			escaped := false

			for j := i + 1; j < len(text) && braceCount > 0; j++ {
				char := text[j]

				if escaped {
					escaped = false
					continue
				}

				if char == '\\' {
					escaped = true
					continue
				}

				if char == '"' {
					inString = !inString
					continue
				}

				if !inString {
					switch char {
					case '{':
						braceCount++
					case '}':
						braceCount--
					}
				}
			}

			if braceCount == 0 {
				// Found complete JSON object
				candidate := text[i : i+findLastBraceIndex(text[i:])+1]
				results = append(results, candidate)
			}
		} else if text[i] == '[' {
			// Find matching closing bracket
			bracketCount := 1
			inString := false
			escaped := false

			for j := i + 1; j < len(text) && bracketCount > 0; j++ {
				char := text[j]

				if escaped {
					escaped = false
					continue
				}

				if char == '\\' {
					escaped = true
					continue
				}

				if char == '"' {
					inString = !inString
					continue
				}

				if !inString {
					switch char {
					case '[':
						bracketCount++
					case ']':
						bracketCount--
					}
				}
			}

			if bracketCount == 0 {
				// Found complete JSON array
				candidate := text[i : i+findLastBracketIndex(text[i:])+1]
				results = append(results, candidate)
			}
		}
	}

	return results
}

// findLastBraceIndex finds the index of the matching closing brace
func findLastBraceIndex(text string) int {
	braceCount := 0
	inString := false
	escaped := false

	for i, char := range text {
		if escaped {
			escaped = false
			continue
		}

		if char == '\\' {
			escaped = true
			continue
		}

		if char == '"' {
			inString = !inString
			continue
		}

		if !inString {
			switch char {
			case '{':
				braceCount++
			case '}':
				braceCount--
				if braceCount == 0 {
					return i
				}
			}
		}
	}

	return -1
}

// findLastBracketIndex finds the index of the matching closing bracket
func findLastBracketIndex(text string) int {
	bracketCount := 0
	inString := false
	escaped := false

	for i, char := range text {
		if escaped {
			escaped = false
			continue
		}

		if char == '\\' {
			escaped = true
			continue
		}

		if char == '"' {
			inString = !inString
			continue
		}

		if !inString {
			switch char {
			case '[':
				bracketCount++
			case ']':
				bracketCount--
				if bracketCount == 0 {
					return i
				}
			}
		}
	}

	return -1
}

// cleanJSON removes comments and cleans up JSON formatting
func cleanJSON(jsonText string) string {
	lines := strings.Split(jsonText, "\n")
	var cleanedLines []string

	for _, line := range lines {
		// Remove line comments
		if idx := strings.Index(line, "//"); idx != -1 {
			line = line[:idx]
		}

		// Remove trailing commas before } or ]
		line = regexp.MustCompile(`,(\s*[}\]])`).ReplaceAllString(line, "$1")

		line = strings.TrimSpace(line)
		if line != "" {
			cleanedLines = append(cleanedLines, line)
		}
	}

	result := strings.Join(cleanedLines, "\n")

	// Basic validation - try to parse as JSON
	var temp interface{}
	if err := json.Unmarshal([]byte(result), &temp); err == nil {
		return result
	}

	return ""
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

// ExtractAndValidateJSON extracts JSON from LLM response and optionally
// validates against a schema
func ExtractAndValidateJSON(response string, schema interface{}) (string, error) {
	jsonStr := ExtractJSONFromResponse(response)

	if schema != nil {
		if err := ValidateAgainstSchema([]byte(jsonStr), schema); err != nil {
			return jsonStr, fmt.Errorf("response validation failed: %w", err)
		}
	}

	return jsonStr, nil
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

// ExtractAndValidateJSONToStruct extracts JSON from LLM response, validates against schema, and unmarshals to struct
func ExtractAndValidateJSONToStruct(response string, out interface{}, schema interface{}) error {
	jsonStr, err := ExtractAndValidateJSON(response, schema)
	if err != nil {
		return err
	}

	return json.Unmarshal([]byte(jsonStr), out)
}
