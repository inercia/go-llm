package llm

import (
	"encoding/json"
	"fmt"

	"github.com/swaggest/jsonschema-go"
)

// SchemaFromStruct generates a JSON Schema from a Go struct using the swaggest/jsonschema-go library
// This provides a Go-idiomatic way to define structured output schemas with full JSON Schema support
//
// Example:
//
//	type Person struct {
//	    Name string `json:"name" jsonschema:"required" description:"Full name"`
//	    Age  int    `json:"age" minimum:"0" maximum:"150"`
//	}
//	schema, err := SchemaFromStruct(Person{})
func SchemaFromStruct(structType interface{}) (interface{}, error) {
	reflector := jsonschema.Reflector{}

	schema, err := reflector.Reflect(structType)
	if err != nil {
		return nil, fmt.Errorf("failed to reflect struct to JSON schema: %w", err)
	}

	return schema, nil
}

// SchemaFromStructAsMap generates a JSON Schema as map[string]interface{} from a Go struct
// This is useful when you need the schema as a generic map for API compatibility
func SchemaFromStructAsMap(structType interface{}) (map[string]interface{}, error) {
	schema, err := SchemaFromStruct(structType)
	if err != nil {
		return nil, err
	}

	// Convert to JSON and back to get a map[string]interface{}
	jsonBytes, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema to JSON: %w", err)
	}

	var schemaMap map[string]interface{}
	if err := json.Unmarshal(jsonBytes, &schemaMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal schema JSON to map: %w", err)
	}

	return schemaMap, nil
}

// ValidateAgainstSchema validates JSON data against a provided JSON Schema
// This uses basic validation - the swaggest library provides schema generation but not validation
// For full validation, consider using github.com/santhosh-tekuri/jsonschema along with this
func ValidateAgainstSchema(data []byte, schema interface{}) error {
	// Parse the data to ensure it's valid JSON
	var parsed interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}

	// TODO: Implement proper JSON Schema validation
	// For now, this just validates the JSON is well-formed
	return nil
}

// NewJSONSchemaResponseFormat creates a ResponseFormat with JSON Schema
func NewJSONSchemaResponseFormat(name, description string, schema interface{}) *ResponseFormat {
	return &ResponseFormat{
		Type: ResponseFormatJSONSchema,
		JSONSchema: &JSONSchema{
			Name:        name,
			Description: description,
			Schema:      schema,
		},
	}
}

// NewJSONSchemaResponseFormatFromStruct creates a ResponseFormat with JSON Schema generated from a Go struct
// This is a convenience method that combines struct reflection and response format creation
func NewJSONSchemaResponseFormatFromStruct(name, description string, structType interface{}) (*ResponseFormat, error) {
	schema, err := SchemaFromStructAsMap(structType)
	if err != nil {
		return nil, fmt.Errorf("failed to generate schema from struct: %w", err)
	}

	return NewJSONSchemaResponseFormat(name, description, schema), nil
}

// NewJSONSchemaResponseFormatStrict creates a ResponseFormat with strict JSON Schema validation
func NewJSONSchemaResponseFormatStrict(name, description string, schema interface{}) *ResponseFormat {
	strict := true
	return &ResponseFormat{
		Type: ResponseFormatJSONSchema,
		JSONSchema: &JSONSchema{
			Name:        name,
			Description: description,
			Schema:      schema,
			Strict:      &strict,
		},
	}
}

// NewJSONSchemaResponseFormatStrictFromStruct creates a strict ResponseFormat with JSON Schema from a Go struct
func NewJSONSchemaResponseFormatStrictFromStruct(name, description string, structType interface{}) (*ResponseFormat, error) {
	schema, err := SchemaFromStructAsMap(structType)
	if err != nil {
		return nil, fmt.Errorf("failed to generate schema from struct: %w", err)
	}

	return NewJSONSchemaResponseFormatStrict(name, description, schema), nil
}

// NewJSONResponseFormat creates a ResponseFormat for basic JSON object output (no schema)
func NewJSONResponseFormat() *ResponseFormat {
	return &ResponseFormat{
		Type: ResponseFormatJSON,
	}
}
