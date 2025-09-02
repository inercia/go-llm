package llm

import (
	"bytes"
	"encoding/json"
	"html/template"

	"github.com/swaggest/jsonschema-go"
)

// PromptsConfig holds prompt configuration with system and user prompts
type PromptsConfig struct {
	System []string `yaml:"system,omitempty"` // System prompts
	User   []string `yaml:"user,omitempty"`   // User prompts
}

// GetSystemPrompts returns all system prompts joined with newlines
func (p PromptsConfig) GetSystemPrompts() string {
	if len(p.System) == 0 {
		return ""
	}
	// Join with newlines
	result := ""
	for i, prompt := range p.System {
		if i > 0 {
			result += "\n"
		}
		result += prompt
	}
	return result
}

// GetUserPrompts returns all user prompts joined with newlines
func (p PromptsConfig) GetUserPrompts() string {
	if len(p.User) == 0 {
		return ""
	}
	// Join with newlines
	result := ""
	for i, prompt := range p.User {
		if i > 0 {
			result += "\n"
		}
		result += prompt
	}
	return result
}

// HasSystemPrompts returns true if there are any system prompts configured
func (p PromptsConfig) HasSystemPrompts() bool {
	return len(p.System) > 0
}

// HasUserPrompts returns true if there are any user prompts configured
func (p PromptsConfig) HasUserPrompts() bool {
	return len(p.User) > 0
}

/////////////////////////////////////////////////////////////////////////////////////////

// PromptTemplate represents a prompt template.
// It can be rendered with specific inputs.
// It uses Go's text/template syntax for placeholders.
type PromptTemplate struct {
	Template string // The prompt template with placeholders
}

// NewPromptTemplate creates a new PromptTemplate with the given template string
func NewPromptTemplate(template string) PromptTemplate {
	return PromptTemplate{
		Template: template,
	}
}

// NewPromptTemplateRendered creates and renders a new PromptTemplate with the given inputs
func NewPromptTemplateRendered(template string, inputs map[string]any) (string, error) {
	pt := PromptTemplate{
		Template: template,
	}
	rendered, err := pt.Render(inputs)
	if err != nil {
		return "", err
	}
	return rendered, nil
}

// Render fills the template with the provided inputs
// It renders using Go's text/template package.
func (pt PromptTemplate) Render(inputs map[string]any) (string, error) {
	tmpl, err := template.New("prompt").Parse(pt.Template)
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, inputs); err != nil {
		return "", err
	}
	return buf.String(), nil
}

// RenderWithJSONSchemaFor fills the template with the provided inputs
// and adds a JSON schema representation of the provided struct 's' under the key "JSONSchema".
// This is useful for prompts that need to include a schema definition.
func (pt PromptTemplate) RenderWithJSONSchemaFor(inputs map[string]any, s any) (string, error) {
	reflector := jsonschema.Reflector{}

	schema, err := reflector.Reflect(s)
	if err != nil {
		return "", err
	}

	j, err := json.MarshalIndent(schema, "", " ")
	if err != nil {
		return "", err
	}

	inputs["JSONSchema"] = string(j)
	return pt.Render(inputs)
}
