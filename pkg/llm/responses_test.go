package llm

import (
	_ "embed"
	"encoding/json"
	"testing"
)

//go:embed fixtures/json-1.json
var json1 string

//go:embed fixtures/json-2.json
var json2 string

//go:embed fixtures/json-3.json
var json3 string

func TestExtractJSONFromResponse(t *testing.T) {
	tests := []struct {
		name     string
		response string
		want     string
	}{
		{
			name:     "json in code block",
			response: "Here is the data:\n```json\n{\"key\": \"value\"}\n```",
			want:     `{"key": "value"}`,
		},
		{
			name:     "json in code block with extra whitespace",
			response: "```json\n   {\"name\": \"John\", \"age\": 30}   \n```",
			want:     `{"name": "John", "age": 30}`,
		},
		{
			name:     "multiline json in code block",
			response: "```json\n{\n  \"users\": [\n    {\"id\": 1, \"name\": \"Alice\"},\n    {\"id\": 2, \"name\": \"Bob\"}\n  ]\n}\n```",
			want:     "{\n  \"users\": [\n    {\"id\": 1, \"name\": \"Alice\"},\n    {\"id\": 2, \"name\": \"Bob\"}\n  ]\n}",
		},
		{
			name:     "json without code block",
			response: "The result is {\"status\": \"success\", \"count\": 5}",
			want:     `{"status": "success", "count": 5}`,
		},
		{
			name:     "json object at beginning",
			response: `{"error": false, "data": {"items": []}} and some other text`,
			want:     `{"error": false, "data": {"items": []}}`,
		},
		{
			name:     "nested json object",
			response: "Response: {\"outer\": {\"inner\": {\"value\": 42}}}",
			want:     `{"outer": {"inner": {"value": 42}}}`,
		},
		{
			name:     "json array",
			response: "Items: [{\"id\": 1}, {\"id\": 2}]",
			want:     "[{\"id\": 1}, {\"id\": 2}]", // Should return the complete valid JSON array
		},
		{
			name:     "empty json object",
			response: "Empty result: {}",
			want:     "{}",
		},
		{
			name:     "json with spaces and newlines",
			response: "Result:\n{\n  \"success\": true,\n  \"message\": \"Operation completed\"\n}",
			want:     "{\n  \"success\": true,\n  \"message\": \"Operation completed\"\n}",
		},
		{
			name:     "multiple json objects - returns first",
			response: "First: {\"a\": 1} and second: {\"b\": 2}",
			want:     `{"a": 1}`, // Should return only the first valid JSON object
		},
		{
			name:     "no json content",
			response: "This is just plain text without any JSON",
			want:     "This is just plain text without any JSON",
		},
		{
			name:     "empty string",
			response: "",
			want:     "",
		},
		{
			name:     "only whitespace",
			response: "   \n\t  ",
			want:     "", // After trimming whitespace, this becomes empty
		},
		{
			name:     "json code block with language case variation - case sensitive",
			response: "```JSON\n{\"test\": true}\n```",
			want:     "{\"test\": true}", // Still extracts because fallback regex finds the JSON object
		},
		{
			name:     "malformed json object",
			response: "Bad JSON: {\"key\": value missing quotes}",
			want:     "Bad JSON: {\"key\": value missing quotes}", // Returns original text since JSON is malformed
		},
		{
			name:     "json with escaped quotes",
			response: "Data: {\"message\": \"Hello \\\"world\\\"\"}",
			want:     `{"message": "Hello \"world\""}`,
		},
		{
			name:     "json in code block with text before and after",
			response: "Before text\n```json\n{\"result\": \"success\"}\n```\nAfter text",
			want:     `{"result": "success"}`,
		},
		{
			name:     "multiple code blocks - returns first json",
			response: "```python\nprint('hello')\n```\n```json\n{\"data\": 1}\n```\n```json\n{\"data\": 2}\n```",
			want:     `{"data": 1}`,
		},
		{
			name:     "complex nested json",
			response: "```json\n{\"users\": [{\"profile\": {\"settings\": {\"theme\": \"dark\"}}}]}\n```",
			want:     `{"users": [{"profile": {"settings": {"theme": "dark"}}}]}`,
		},
		{
			name:     "code block without language specifier",
			response: "Here's the result:\n```\n{\"success\": true, \"data\": [1, 2, 3]}\n```",
			want:     `{"success": true, "data": [1, 2, 3]}`,
		},
		{
			name:     "code block with javascript language",
			response: "```javascript\n{\"api\": \"response\", \"status\": \"ok\"}\n```",
			want:     `{"api": "response", "status": "ok"}`,
		},
		{
			name:     "code block with js language",
			response: "```js\n{\"result\": {\"count\": 42}}\n```",
			want:     `{"result": {"count": 42}}`,
		},
		{
			name:     "code block mixed with non-json content",
			response: "```python\nprint('hello')\n```\n\nThen:\n```json\n{\"extracted\": true}\n```",
			want:     `{"extracted": true}`,
		},
		{
			name:     "deeply nested json object",
			response: "Complex: {\"level1\": {\"level2\": {\"level3\": {\"level4\": {\"value\": \"deep\"}}}}}",
			want:     `{"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}`,
		},
		{
			name:     "json array with nested objects",
			response: "Results: [{\"id\": 1, \"meta\": {\"tag\": \"a\"}}, {\"id\": 2, \"meta\": {\"tag\": \"b\"}}]",
			want:     `[{"id": 1, "meta": {"tag": "a"}}, {"id": 2, "meta": {"tag": "b"}}]`,
		},
		{
			name:     "json with special characters and escapes",
			response: "```json\n{\"path\": \"/home/user\\\\file.txt\", \"query\": \"SELECT * FROM \\\"table\\\"\"}\n```",
			want:     `{"path": "/home/user\\file.txt", "query": "SELECT * FROM \"table\""}`,
		},
		{
			name:     "json fixture with some prefixed text",
			response: json1,
			want:     "{\n    \"name\": \"John\",\n    \"age\": 30\n}",
		},
		{
			name:     "json fixture with some prefixed text",
			response: json2,
			want:     "{\n    \"name\": \"John\",\n    \"age\": 40\n}",
		},
		{
			name:     "json fixture not in a code block",
			response: json3,
			want:     "{\n    \"name\": \"John\",\n    \"age\": 50\n}",
		},
		{
			name: "real-world analysis response with json output",
			response: `**Analysis Insights**

1. **Account Lockdown**: The account has been temporarily locked due to multiple failed login attempts.
2. **Security Concerns**: The security reason for the lockdown is unclear, but it's likely related to the attempted password combinations.
3. **Urgent Need**: Sarah Wilson requires access to her account for an important client meeting this afternoon.

**Issue Type Categorization**

Based on the analysis, I categorize the issue as follows:

* **Account-Security**: The account has been locked due to security reasons, which is a clear indication of a security concern.
* **General-Guidelines**: Sarah Wilson's request for assistance and additional verification falls under general guidelines.

**Priority Categorization**

Given the urgency of the situation, I categorize the priority as:

* **P0-Critical**: The account lockdown affects Sarah Wilson's ability to perform her job duties, which is critical to her work.

**Communication Style**

Based on the email content, I would categorize the communication style as:

* **Polite**: Although the tone of the email is professional, there is a hint of urgency and concern for the situation.

**Subscription Tier**

Unfortunately, the provided information does not explicitly mention Sarah Wilson's subscription tier. However, based on the context, it can be inferred that she likely has an Enterprise or Professional subscription, given the importance of her work.

**JSON Output**

Here is the JSON output with the analyzed insights:

` + "```json\n" + `{
  "from_email": "sarah.wilson@consulting.com",
  "from_name": "Sarah Wilson",
  "customer_id": "CUST-67890",
  "issue_type": ["account-security", "general-guidelines"],
  "priority": "P0-Critical",
  "communication_style": "Polite",
  "subscription_tier": "Enterprise"
}
` + "```" + `

Please note that the subscription tier is inferred based on the context, and it's essential to verify this information with the primary agent or relevant authorities.`,
			want: `{
  "from_email": "sarah.wilson@consulting.com",
  "from_name": "Sarah Wilson",
  "customer_id": "CUST-67890",
  "issue_type": ["account-security", "general-guidelines"],
  "priority": "P0-Critical",
  "communication_style": "Polite",
  "subscription_tier": "Enterprise"
}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExtractJSONFromResponse(tt.response)
			if got != tt.want {
				t.Errorf("ExtractJSONFromResponse() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestExtractJSONToStruct(t *testing.T) {
	// Define test structs
	type Person struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}

	type Response struct {
		Status  string   `json:"status"`
		Data    []Person `json:"data"`
		Success bool     `json:"success"`
	}

	type NestedData struct {
		Outer struct {
			Inner struct {
				Value int `json:"value"`
			} `json:"inner"`
		} `json:"outer"`
	}

	tests := []struct {
		name          string
		response      string
		target        interface{}
		expectError   bool
		expectedValue interface{}
	}{
		{
			name:     "simple struct from json code block",
			response: "```json\n{\"name\": \"John\", \"age\": 30}\n```",
			target:   &Person{},
			expectedValue: &Person{
				Name: "John",
				Age:  30,
			},
		},
		{
			name:     "complex struct from json code block",
			response: "```json\n{\"status\": \"ok\", \"data\": [{\"name\": \"Alice\", \"age\": 25}], \"success\": true}\n```",
			target:   &Response{},
			expectedValue: &Response{
				Status:  "ok",
				Data:    []Person{{Name: "Alice", Age: 25}},
				Success: true,
			},
		},
		{
			name:     "nested struct",
			response: "{\"outer\": {\"inner\": {\"value\": 42}}}",
			target:   &NestedData{},
			expectedValue: &NestedData{
				Outer: struct {
					Inner struct {
						Value int `json:"value"`
					} `json:"inner"`
				}{
					Inner: struct {
						Value int `json:"value"`
					}{
						Value: 42,
					},
				},
			},
		},
		{
			name:     "empty object",
			response: "{}",
			target:   &Person{},
			expectedValue: &Person{
				Name: "",
				Age:  0,
			},
		},
		{
			name:        "invalid json",
			response:    "{\"name\": John, \"age\": 30}", // missing quotes around John
			target:      &Person{},
			expectError: true,
		},
		{
			name:        "non-json text",
			response:    "This is not JSON at all",
			target:      &Person{},
			expectError: true,
		},
		{
			name:        "empty string",
			response:    "",
			target:      &Person{},
			expectError: true,
		},
		{
			name:     "json with extra fields",
			response: "{\"name\": \"Bob\", \"age\": 35, \"email\": \"bob@example.com\"}",
			target:   &Person{},
			expectedValue: &Person{
				Name: "Bob",
				Age:  35,
			},
		},
		{
			name:     "json array to slice",
			response: "```json\n[{\"name\": \"Alice\", \"age\": 25}, {\"name\": \"Bob\", \"age\": 30}]\n```",
			target:   &[]Person{},
			expectedValue: &[]Person{
				{Name: "Alice", Age: 25},
				{Name: "Bob", Age: 30},
			},
		},
		{
			name:     "partial json match",
			response: "The result is {\"name\": \"Charlie\", \"age\": 40} and some other text",
			target:   &Person{},
			expectedValue: &Person{
				Name: "Charlie",
				Age:  40,
			},
		},
		{
			name: "real-world analysis response",
			response: `**Analysis Insights**

1. **Account Lockdown**: The account has been temporarily locked due to multiple failed login attempts.

**JSON Output**

Here is the JSON output with the analyzed insights:

` + "```json\n" + `{
  "from_email": "sarah.wilson@consulting.com",
  "from_name": "Sarah Wilson",
  "customer_id": "CUST-67890",
  "issue_type": ["account-security", "general-guidelines"],
  "priority": "P0-Critical",
  "communication_style": "Polite",
  "subscription_tier": "Enterprise"
}
` + "```" + `

Please note that the subscription tier is inferred based on the context.`,
			target: &struct {
				FromEmail          string   `json:"from_email"`
				FromName           string   `json:"from_name"`
				CustomerID         string   `json:"customer_id"`
				IssueType          []string `json:"issue_type"`
				Priority           string   `json:"priority"`
				CommunicationStyle string   `json:"communication_style"`
				SubscriptionTier   string   `json:"subscription_tier"`
			}{},
			expectedValue: &struct {
				FromEmail          string   `json:"from_email"`
				FromName           string   `json:"from_name"`
				CustomerID         string   `json:"customer_id"`
				IssueType          []string `json:"issue_type"`
				Priority           string   `json:"priority"`
				CommunicationStyle string   `json:"communication_style"`
				SubscriptionTier   string   `json:"subscription_tier"`
			}{
				FromEmail:          "sarah.wilson@consulting.com",
				FromName:           "Sarah Wilson",
				CustomerID:         "CUST-67890",
				IssueType:          []string{"account-security", "general-guidelines"},
				Priority:           "P0-Critical",
				CommunicationStyle: "Polite",
				SubscriptionTier:   "Enterprise",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ExtractJSONToStruct(tt.response, tt.target)

			if tt.expectError {
				if err == nil {
					t.Errorf("ExtractJSONToStruct() expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("ExtractJSONToStruct() unexpected error: %v", err)
				return
			}

			// Compare the results
			expectedJSON, err := json.Marshal(tt.expectedValue)
			if err != nil {
				t.Fatalf("Failed to marshal expected value: %v", err)
			}

			actualJSON, err := json.Marshal(tt.target)
			if err != nil {
				t.Fatalf("Failed to marshal actual value: %v", err)
			}

			if string(expectedJSON) != string(actualJSON) {
				t.Errorf("ExtractJSONToStruct() = %s, want %s", string(actualJSON), string(expectedJSON))
			}
		})
	}
}

// Test edge cases and error scenarios
func TestExtractJSONToStruct_EdgeCases(t *testing.T) {
	t.Run("nil target", func(t *testing.T) {
		// json.Unmarshal with nil target should return an error, not panic
		err := ExtractJSONToStruct("{\"test\": true}", nil)
		if err == nil {
			t.Errorf("ExtractJSONToStruct() with nil target should return error")
		}
	})

	t.Run("non-pointer target", func(t *testing.T) {
		type TestStruct struct {
			Test bool `json:"test"`
		}
		var target TestStruct
		err := ExtractJSONToStruct("{\"test\": true}", target) // passing value instead of pointer
		if err == nil {
			t.Errorf("ExtractJSONToStruct() with non-pointer target should return error")
		}
	})

	t.Run("json type mismatch", func(t *testing.T) {
		type TestStruct struct {
			Age int `json:"age"`
		}
		var target TestStruct
		err := ExtractJSONToStruct("{\"age\": \"not a number\"}", &target)
		if err == nil {
			t.Errorf("ExtractJSONToStruct() with type mismatch should return error")
		}
	})
}

// Benchmark tests
func BenchmarkExtractJSONFromResponse_CodeBlock(b *testing.B) {
	response := "Here is the data:\n```json\n{\"key\": \"value\", \"number\": 42, \"array\": [1, 2, 3]}\n```"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ExtractJSONFromResponse(response)
	}
}

func BenchmarkExtractJSONFromResponse_PlainJSON(b *testing.B) {
	response := "Result: {\"key\": \"value\", \"number\": 42, \"array\": [1, 2, 3]}"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ExtractJSONFromResponse(response)
	}
}

func BenchmarkExtractJSONToStruct(b *testing.B) {
	type TestStruct struct {
		Key    string `json:"key"`
		Number int    `json:"number"`
		Array  []int  `json:"array"`
	}
	response := "```json\n{\"key\": \"value\", \"number\": 42, \"array\": [1, 2, 3]}\n```"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var target TestStruct
		_ = ExtractJSONToStruct(response, &target)
	}
}

func TestRemoveBlocks(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		tag      string
		expected string
	}{
		{
			name:     "simple single block",
			text:     "Hello <think>this is internal</think> world!",
			tag:      "think",
			expected: "Hello  world!",
		},
		{
			name:     "multiple blocks same tag",
			text:     "Start <think>first thought</think> middle <think>second thought</think> end",
			tag:      "think",
			expected: "Start  middle  end",
		},
		{
			name:     "multiline block content",
			text:     "Before <think>\nThis is a\nmultiline thought\nwith newlines\n</think> After",
			tag:      "think",
			expected: "Before  After",
		},
		{
			name:     "empty block",
			text:     "Hello <think></think> world!",
			tag:      "think",
			expected: "Hello  world!",
		},
		{
			name:     "no blocks present",
			text:     "This text has no special blocks at all.",
			tag:      "think",
			expected: "This text has no special blocks at all.",
		},
		{
			name:     "block at beginning",
			text:     "<think>initial thought</think> rest of text",
			tag:      "think",
			expected: " rest of text",
		},
		{
			name:     "block at end",
			text:     "main content <think>final thought</think>",
			tag:      "think",
			expected: "main content ",
		},
		{
			name:     "only block content",
			text:     "<think>just thinking</think>",
			tag:      "think",
			expected: "",
		},
		{
			name:     "different tag name",
			text:     "Hello <debug>debug info here</debug> world!",
			tag:      "debug",
			expected: "Hello  world!",
		},
		{
			name:     "wrong tag name - no removal",
			text:     "Hello <think>this should remain</think> world!",
			tag:      "debug",
			expected: "Hello <think>this should remain</think> world!",
		},
		{
			name:     "case sensitive tag",
			text:     "Hello <Think>this should remain</Think> world!",
			tag:      "think",
			expected: "Hello <Think>this should remain</Think> world!",
		},
		{
			name:     "special characters in tag",
			text:     "Hello <my-tag>content</my-tag> world!",
			tag:      "my-tag",
			expected: "Hello  world!",
		},
		{
			name:     "special regex characters in tag",
			text:     "Hello <test+tag>content</test+tag> world!",
			tag:      "test+tag",
			expected: "Hello  world!",
		},
		{
			name:     "blocks with attributes - should not match",
			text:     "Hello <think class='test'>content</think> world!",
			tag:      "think",
			expected: "Hello <think class='test'>content</think> world!",
		},
		{
			name:     "malformed blocks - unclosed",
			text:     "Hello <think>unclosed block world!",
			tag:      "think",
			expected: "Hello <think>unclosed block world!",
		},
		{
			name:     "malformed blocks - wrong closing tag",
			text:     "Hello <think>content</debug> world!",
			tag:      "think",
			expected: "Hello <think>content</debug> world!",
		},
		{
			name:     "nested content but different tags",
			text:     "Hello <think>outer <debug>inner</debug> content</think> world!",
			tag:      "think",
			expected: "Hello  world!",
		},
		{
			name:     "nested same tags - removes outer only",
			text:     "Hello <think>outer <think>inner</think> content</think> world!",
			tag:      "think",
			expected: "Hello  content</think> world!", // Regex matches first opening to first closing
		},
		{
			name:     "block with complex content",
			text:     "Start <think>This has: symbols, numbers 123, and\n\"quoted strings\" with 'apostrophes'</think> End",
			tag:      "think",
			expected: "Start  End",
		},
		{
			name:     "multiple different tags mixed",
			text:     "Hello <think>thought</think> and <debug>info</debug> world!",
			tag:      "think",
			expected: "Hello  and <debug>info</debug> world!",
		},
		{
			name:     "empty string",
			text:     "",
			tag:      "think",
			expected: "",
		},
		{
			name:     "empty tag name",
			text:     "Hello <think>content</think> world!",
			tag:      "",
			expected: "Hello <think>content</think> world!",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := RemoveBlocks(tt.text, tt.tag)
			if result != tt.expected {
				t.Errorf("RemoveBlocks() = %q, expected %q", result, tt.expected)
			}
		})
	}
}

// Benchmark tests
func BenchmarkRemoveBlocks(b *testing.B) {
	text := `This is a sample text with <think>some internal thoughts that should be removed</think> and normal content that should remain. Here's another <think>block with more complex content including:
- Special characters: !@#$%^&*()
- Numbers: 123456789
- Multiple lines of text
- And even more content</think> at the end.`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = RemoveBlocks(text, "think")
	}
}
