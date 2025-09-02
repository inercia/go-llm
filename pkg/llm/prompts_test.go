package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPromptsConfig_GetSystemPrompts(t *testing.T) {
	tests := []struct {
		name     string
		config   PromptsConfig
		expected string
	}{
		{
			name:     "empty system prompts",
			config:   PromptsConfig{System: nil},
			expected: "",
		},
		{
			name:     "single system prompt",
			config:   PromptsConfig{System: []string{"You are a helpful assistant."}},
			expected: "You are a helpful assistant.",
		},
		{
			name: "multiple system prompts",
			config: PromptsConfig{System: []string{
				"You are a helpful assistant.",
				"Always be polite.",
				"Provide accurate information.",
			}},
			expected: "You are a helpful assistant.\nAlways be polite.\nProvide accurate information.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.config.GetSystemPrompts()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestPromptsConfig_GetUserPrompts(t *testing.T) {
	tests := []struct {
		name     string
		config   PromptsConfig
		expected string
	}{
		{
			name:     "empty user prompts",
			config:   PromptsConfig{User: nil},
			expected: "",
		},
		{
			name:     "single user prompt",
			config:   PromptsConfig{User: []string{"What is the weather today?"}},
			expected: "What is the weather today?",
		},
		{
			name: "multiple user prompts",
			config: PromptsConfig{User: []string{
				"What is the weather today?",
				"Can you help me with coding?",
				"Tell me a joke.",
			}},
			expected: "What is the weather today?\nCan you help me with coding?\nTell me a joke.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.config.GetUserPrompts()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestPromptsConfig_HasSystemPrompts(t *testing.T) {
	tests := []struct {
		name     string
		config   PromptsConfig
		expected bool
	}{
		{
			name:     "no system prompts",
			config:   PromptsConfig{System: nil},
			expected: false,
		},
		{
			name:     "empty system prompts slice",
			config:   PromptsConfig{System: []string{}},
			expected: false,
		},
		{
			name:     "has system prompts",
			config:   PromptsConfig{System: []string{"You are a helpful assistant."}},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.config.HasSystemPrompts()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestPromptsConfig_HasUserPrompts(t *testing.T) {
	tests := []struct {
		name     string
		config   PromptsConfig
		expected bool
	}{
		{
			name:     "no user prompts",
			config:   PromptsConfig{User: nil},
			expected: false,
		},
		{
			name:     "empty user prompts slice",
			config:   PromptsConfig{User: []string{}},
			expected: false,
		},
		{
			name:     "has user prompts",
			config:   PromptsConfig{User: []string{"What is the weather?"}},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.config.HasUserPrompts()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestNewPromptTemplate(t *testing.T) {
	template := "Hello {{.Name}}, how are you?"
	pt := NewPromptTemplate(template)

	assert.Equal(t, template, pt.Template)
}

func TestNewPromptTemplateRendered(t *testing.T) {
	tests := []struct {
		name     string
		template string
		inputs   map[string]any
		expected string
		hasError bool
	}{
		{
			name:     "simple template",
			template: "Hello {{.Name}}!",
			inputs:   map[string]any{"Name": "World"},
			expected: "Hello World!",
			hasError: false,
		},
		{
			name:     "multiple variables",
			template: "Hello {{.Name}}, you are {{.Age}} years old.",
			inputs:   map[string]any{"Name": "Alice", "Age": 30},
			expected: "Hello Alice, you are 30 years old.",
			hasError: false,
		},
		{
			name:     "no variables",
			template: "This is a static template.",
			inputs:   map[string]any{},
			expected: "This is a static template.",
			hasError: false,
		},
		{
			name:     "invalid template syntax",
			template: "Hello {{.Name",
			inputs:   map[string]any{"Name": "World"},
			expected: "",
			hasError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := NewPromptTemplateRendered(tt.template, tt.inputs)

			if tt.hasError {
				assert.Error(t, err)
				assert.Equal(t, tt.expected, result)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}

func TestPromptTemplate_Render(t *testing.T) {
	tests := []struct {
		name     string
		template string
		inputs   map[string]any
		expected string
		hasError bool
	}{
		{
			name:     "simple variable substitution",
			template: "Hello {{.Name}}!",
			inputs:   map[string]any{"Name": "World"},
			expected: "Hello World!",
			hasError: false,
		},
		{
			name:     "multiple variables",
			template: "User: {{.User}}, Action: {{.Action}}, Status: {{.Status}}",
			inputs:   map[string]any{"User": "john", "Action": "login", "Status": "success"},
			expected: "User: john, Action: login, Status: success",
			hasError: false,
		},
		{
			name:     "nested object access",
			template: "Hello {{.User.Name}}, your ID is {{.User.ID}}",
			inputs: map[string]any{
				"User": map[string]any{
					"Name": "Alice",
					"ID":   123,
				},
			},
			expected: "Hello Alice, your ID is 123",
			hasError: false,
		},
		{
			name:     "conditional rendering",
			template: "{{if .IsAdmin}}Welcome, admin!{{else}}Welcome, user!{{end}}",
			inputs:   map[string]any{"IsAdmin": true},
			expected: "Welcome, admin!",
			hasError: false,
		},
		{
			name:     "range over slice",
			template: "Items: {{range .Items}}{{.}} {{end}}",
			inputs:   map[string]any{"Items": []string{"apple", "banana", "cherry"}},
			expected: "Items: apple banana cherry ",
			hasError: false,
		},
		{
			name:     "empty template",
			template: "",
			inputs:   map[string]any{"Name": "World"},
			expected: "",
			hasError: false,
		},
		{
			name:     "template with functions",
			template: "Name: {{.Name | printf \"%q\"}}",
			inputs:   map[string]any{"Name": "John"},
			expected: "Name: &#34;John&#34;", // HTML escaped quotes
			hasError: false,
		},
		{
			name:     "missing variable",
			template: "Hello {{.Name}}!",
			inputs:   map[string]any{},
			expected: "Hello !", // html/template shows empty string for missing values
			hasError: false,
		},
		{
			name:     "invalid template syntax",
			template: "Hello {{.Name",
			inputs:   map[string]any{"Name": "World"},
			expected: "",
			hasError: true,
		},
		{
			name:     "invalid variable access",
			template: "Hello {{.User.InvalidField}}!",
			inputs:   map[string]any{"User": map[string]any{"Name": "Alice"}},
			expected: "Hello !", // html/template shows empty string for missing fields
			hasError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pt := PromptTemplate{Template: tt.template}
			result, err := pt.Render(tt.inputs)

			if tt.hasError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}

func TestPromptTemplate_RenderWithJSONSchemaFor(t *testing.T) {
	// Define a test struct for schema generation
	type TestStruct struct {
		Name string `json:"name" description:"The name of the person"`
		Age  int    `json:"age" description:"The age of the person"`
	}

	tests := []struct {
		name         string
		template     string
		inputs       map[string]any
		schemaStruct any
		hasError     bool
		checkSchema  bool
	}{
		{
			name:         "simple template with JSON schema",
			template:     "Hello {{.Name}}! Schema: {{.JSONSchema}}",
			inputs:       map[string]any{"Name": "World"},
			schemaStruct: TestStruct{},
			hasError:     false,
			checkSchema:  true,
		},
		{
			name:         "template without using schema",
			template:     "Hello {{.Name}}!",
			inputs:       map[string]any{"Name": "World"},
			schemaStruct: TestStruct{},
			hasError:     false,
			checkSchema:  false,
		},
		{
			name:         "invalid struct for schema",
			template:     "Schema: {{.JSONSchema}}",
			inputs:       map[string]any{},
			schemaStruct: make(chan int), // Invalid type for JSON schema
			hasError:     true,
			checkSchema:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pt := PromptTemplate{Template: tt.template}
			result, err := pt.RenderWithJSONSchemaFor(tt.inputs, tt.schemaStruct)

			if tt.hasError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)

				if tt.checkSchema {
					// Verify that JSONSchema was added to the result
					// Note: html/template escapes quotes as &#34;
					assert.Contains(t, result, `&#34;type&#34;: &#34;object&#34;`)
					assert.Contains(t, result, `&#34;properties&#34;`)

					// For our TestStruct, we should see name and age properties
					assert.Contains(t, result, `&#34;name&#34;`)
					assert.Contains(t, result, `&#34;age&#34;`)
				}
			}
		})
	}
}

func TestPromptTemplate_RenderWithJSONSchemaFor_OverridesJSONSchema(t *testing.T) {
	// Test that RenderWithJSONSchemaFor overrides any existing JSONSchema in inputs
	type SimpleStruct struct {
		Field string `json:"field"`
	}

	pt := PromptTemplate{Template: "Schema: {{.JSONSchema}}"}
	inputs := map[string]any{
		"JSONSchema": "This should be overridden",
	}

	result, err := pt.RenderWithJSONSchemaFor(inputs, SimpleStruct{})
	require.NoError(t, err)

	// Should contain the generated schema, not the original value
	// Note: html/template escapes quotes as &#34;
	assert.Contains(t, result, `&#34;type&#34;: &#34;object&#34;`)
	assert.Contains(t, result, `&#34;field&#34;`)
	assert.NotContains(t, result, "This should be overridden")
}

func TestPromptTemplate_ComplexExample(t *testing.T) {
	// Test a complex real-world example
	template := `You are an AI assistant helping with {{.Task}}.

User Information:
- Name: {{.User.Name}}
- Role: {{.User.Role}}
- Permissions: {{range .User.Permissions}}{{.}} {{end}}

{{if .IncludeSchema}}
Please respond according to this JSON schema:
{{.JSONSchema}}
{{end}}

Current task: {{.Task}}
Priority: {{.Priority | printf "%q"}}
`

	type ResponseSchema struct {
		Success bool   `json:"success" description:"Whether the operation was successful"`
		Message string `json:"message" description:"Response message"`
	}

	inputs := map[string]any{
		"Task":     "data analysis",
		"Priority": "high",
		"User": map[string]any{
			"Name":        "Alice Johnson",
			"Role":        "Data Scientist",
			"Permissions": []string{"read", "write", "analyze"},
		},
		"IncludeSchema": true,
	}

	pt := PromptTemplate{Template: template}
	result, err := pt.RenderWithJSONSchemaFor(inputs, ResponseSchema{})
	require.NoError(t, err)

	// Verify all components are present
	assert.Contains(t, result, "helping with data analysis")
	assert.Contains(t, result, "Alice Johnson")
	assert.Contains(t, result, "Data Scientist")
	assert.Contains(t, result, "read write analyze")
	assert.Contains(t, result, "Priority: &#34;high&#34;")         // HTML escaped quotes
	assert.Contains(t, result, `&#34;type&#34;: &#34;object&#34;`) // HTML escaped quotes
	assert.Contains(t, result, `&#34;success&#34;`)
	assert.Contains(t, result, `&#34;message&#34;`)
}
