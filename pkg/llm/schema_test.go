package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test structures for schema generation
type SimplePerson struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

type PersonWithValidation struct {
	Name     string  `json:"name" description:"Person's full name" minLength:"1" maxLength:"100"`
	Age      int     `json:"age" minimum:"0" maximum:"150" description:"Person's age in years"`
	Email    string  `json:"email,omitempty" format:"email"`
	Salary   float64 `json:"salary,omitempty" minimum:"0"`
	IsActive bool    `json:"is_active"`
}

type NestedStruct struct {
	Person SimplePerson `json:"person"`
	Items  []string     `json:"items"`
	Count  int          `json:"count"`
}

func TestSchemaFromStruct(t *testing.T) {
	tests := []struct {
		name       string
		structType interface{}
		wantErr    bool
	}{
		{
			name:       "simple struct",
			structType: SimplePerson{},
			wantErr:    false,
		},
		{
			name:       "struct with validation tags",
			structType: PersonWithValidation{},
			wantErr:    false,
		},
		{
			name:       "nested struct",
			structType: NestedStruct{},
			wantErr:    false,
		},
		{
			name:       "pointer to struct",
			structType: &SimplePerson{},
			wantErr:    false,
		},
		{
			name:       "non-struct type",
			structType: "not a struct",
			wantErr:    false, // swaggest library handles non-structs gracefully
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			schema, err := SchemaFromStruct(tt.structType)

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.NotNil(t, schema)
		})
	}
}

func TestSchemaFromStructAsMap(t *testing.T) {
	tests := []struct {
		name       string
		structType interface{}
		wantErr    bool
		validate   func(t *testing.T, schema map[string]interface{})
	}{
		{
			name:       "simple person struct",
			structType: SimplePerson{},
			wantErr:    false,
			validate: func(t *testing.T, schema map[string]interface{}) {
				assert.Equal(t, "object", schema["type"])

				properties, ok := schema["properties"].(map[string]interface{})
				require.True(t, ok, "properties should be a map")

				// Check name field
				nameField, ok := properties["name"].(map[string]interface{})
				require.True(t, ok, "name field should exist")
				assert.Equal(t, "string", nameField["type"])

				// Check age field
				ageField, ok := properties["age"].(map[string]interface{})
				require.True(t, ok, "age field should exist")
				assert.Equal(t, "integer", ageField["type"])
			},
		},
		{
			name:       "person with validation",
			structType: PersonWithValidation{},
			wantErr:    false,
			validate: func(t *testing.T, schema map[string]interface{}) {
				assert.Equal(t, "object", schema["type"])

				properties, ok := schema["properties"].(map[string]interface{})
				require.True(t, ok, "properties should be a map")

				// Check if validation constraints are applied
				nameField, ok := properties["name"].(map[string]interface{})
				require.True(t, ok, "name field should exist")
				assert.Equal(t, "string", nameField["type"])
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			schema, err := SchemaFromStructAsMap(tt.structType)

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.NotNil(t, schema)

			if tt.validate != nil {
				tt.validate(t, schema)
			}
		})
	}
}

func TestResponseFormatCreation(t *testing.T) {
	t.Run("NewJSONSchemaResponseFormat", func(t *testing.T) {
		schema := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{"type": "string"},
			},
		}

		rf := NewJSONSchemaResponseFormat("test_schema", "A test schema", schema)

		assert.Equal(t, ResponseFormatJSONSchema, rf.Type)
		assert.NotNil(t, rf.JSONSchema)
		assert.Equal(t, "test_schema", rf.JSONSchema.Name)
		assert.Equal(t, "A test schema", rf.JSONSchema.Description)
		assert.Equal(t, schema, rf.JSONSchema.Schema)
		assert.Nil(t, rf.JSONSchema.Strict)
	})

	t.Run("NewJSONSchemaResponseFormatStrict", func(t *testing.T) {
		schema := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{"type": "string"},
			},
		}

		rf := NewJSONSchemaResponseFormatStrict("test_schema", "A test schema", schema)

		assert.Equal(t, ResponseFormatJSONSchema, rf.Type)
		assert.NotNil(t, rf.JSONSchema)
		assert.Equal(t, "test_schema", rf.JSONSchema.Name)
		assert.Equal(t, "A test schema", rf.JSONSchema.Description)
		assert.Equal(t, schema, rf.JSONSchema.Schema)
		assert.NotNil(t, rf.JSONSchema.Strict)
		assert.True(t, *rf.JSONSchema.Strict)
	})

	t.Run("NewJSONSchemaResponseFormatFromStruct", func(t *testing.T) {
		rf, err := NewJSONSchemaResponseFormatFromStruct("person", "A person object", SimplePerson{})

		require.NoError(t, err)
		assert.Equal(t, ResponseFormatJSONSchema, rf.Type)
		assert.NotNil(t, rf.JSONSchema)
		assert.Equal(t, "person", rf.JSONSchema.Name)
		assert.Equal(t, "A person object", rf.JSONSchema.Description)
		assert.NotNil(t, rf.JSONSchema.Schema)
	})

	t.Run("NewJSONResponseFormat", func(t *testing.T) {
		rf := NewJSONResponseFormat()

		assert.Equal(t, ResponseFormatJSON, rf.Type)
		assert.Nil(t, rf.JSONSchema)
	})
}

func TestValidateAgainstSchema(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{"type": "string"},
			"age":  map[string]interface{}{"type": "integer"},
		},
		"required": []string{"name", "age"},
	}

	tests := []struct {
		name     string
		jsonData string
		schema   interface{}
		wantErr  bool
	}{
		{
			name:     "valid JSON with schema",
			jsonData: `{"name": "John", "age": 30}`,
			schema:   schema,
			wantErr:  false,
		},
		{
			name:     "invalid JSON",
			jsonData: `{"name": "John", "age":}`,
			schema:   schema,
			wantErr:  true,
		},
		{
			name:     "valid JSON without schema",
			jsonData: `{"name": "John", "age": 30}`,
			schema:   nil,
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateAgainstSchema([]byte(tt.jsonData), tt.schema)

			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// Test the new response processing functions
func TestExtractAndValidateJSON(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{"type": "string"},
		},
	}

	tests := []struct {
		name     string
		response string
		schema   interface{}
		want     string
		wantErr  bool
	}{
		{
			name:     "valid JSON in markdown",
			response: "```json\n{\"name\": \"John\"}\n```",
			schema:   schema,
			want:     `{"name": "John"}`,
			wantErr:  false,
		},
		{
			name:     "valid JSON without schema",
			response: "```json\n{\"name\": \"John\"}\n```",
			schema:   nil,
			want:     `{"name": "John"}`,
			wantErr:  false,
		},
		{
			name:     "invalid JSON with schema",
			response: "```json\n{\"name\":}\n```",
			schema:   schema,
			want:     `{"name":}`,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ExtractAndValidateJSON(tt.response, tt.schema)

			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, result)
			}
		})
	}
}

func TestExtractAndValidateJSONToStruct(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{"type": "string"},
			"age":  map[string]interface{}{"type": "integer"},
		},
	}

	tests := []struct {
		name     string
		response string
		schema   interface{}
		want     SimplePerson
		wantErr  bool
	}{
		{
			name:     "valid JSON to struct",
			response: "```json\n{\"name\": \"John\", \"age\": 30}\n```",
			schema:   schema,
			want:     SimplePerson{Name: "John", Age: 30},
			wantErr:  false,
		},
		{
			name:     "invalid JSON to struct",
			response: "```json\n{\"name\": \"John\", \"age\":}\n```",
			schema:   schema,
			want:     SimplePerson{},
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var result SimplePerson
			err := ExtractAndValidateJSONToStruct(tt.response, &result, tt.schema)

			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, result)
			}
		})
	}
}

// Benchmark tests
func BenchmarkSchemaFromStruct(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = SchemaFromStruct(PersonWithValidation{})
	}
}

func BenchmarkSchemaFromStructAsMap(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = SchemaFromStructAsMap(PersonWithValidation{})
	}
}
