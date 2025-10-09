package bedrock

import (
	"testing"

	"github.com/inercia/go-llm/pkg/llm"
)

func TestNewClient(t *testing.T) {
	tests := []struct {
		name    string
		config  llm.ClientConfig
		wantErr bool
	}{
		{
			name: "valid config with default region",
			config: llm.ClientConfig{
				Provider: "bedrock",
				Model:    "anthropic.claude-3-haiku-20240307-v1:0",
			},
			wantErr: false,
		},
		{
			name: "valid config with custom region",
			config: llm.ClientConfig{
				Provider: "bedrock",
				Model:    "anthropic.claude-3-haiku-20240307-v1:0",
				Extra: map[string]string{
					"region": "us-west-2",
				},
			},
			wantErr: false,
		},
		{
			name: "valid config with custom endpoints",
			config: llm.ClientConfig{
				Provider: "bedrock",
				Model:    "anthropic.claude-3-haiku-20240307-v1:0",
				Extra: map[string]string{
					"region":                   "us-west-2",
					"bedrock_endpoint":         "https://bedrock.custom.amazonaws.com",
					"bedrock_runtime_endpoint": "https://bedrock-runtime.custom.amazonaws.com",
				},
			},
			wantErr: false,
		},
		{
			name: "valid config with BaseURL",
			config: llm.ClientConfig{
				Provider: "bedrock",
				Model:    "anthropic.claude-3-haiku-20240307-v1:0",
				BaseURL:  "https://bedrock-runtime.custom.amazonaws.com",
				Extra: map[string]string{
					"region": "us-west-2",
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewClient(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewClient() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && client == nil {
				t.Errorf("NewClient() returned nil client")
				return
			}
			if client != nil {
				if client.model != tt.config.Model {
					t.Errorf("NewClient() model = %v, want %v", client.model, tt.config.Model)
				}
				if client.provider != "bedrock" {
					t.Errorf("NewClient() provider = %v, want bedrock", client.provider)
				}
			}
		})
	}
}

func TestGetModelInfo(t *testing.T) {
	client := &Client{
		model:    "anthropic.claude-3-sonnet-20240229-v1:0",
		provider: "bedrock",
	}

	modelInfo := client.GetModelInfo()

	if modelInfo.Name != client.model {
		t.Errorf("GetModelInfo() Name = %v, want %v", modelInfo.Name, client.model)
	}
	if modelInfo.Provider != "bedrock" {
		t.Errorf("GetModelInfo() Provider = %v, want bedrock", modelInfo.Provider)
	}
	if !modelInfo.SupportsStreaming {
		t.Errorf("GetModelInfo() SupportsStreaming = false, want true")
	}
}

func TestModelDetection(t *testing.T) {
	tests := []struct {
		model      string
		wantClaude bool
		wantTitan  bool
		wantLlama  bool
		wantVision bool
		wantTools  bool
	}{
		{
			model:      "anthropic.claude-3-sonnet-20240229-v1:0",
			wantClaude: true,
			wantVision: true,
			wantTools:  true,
		},
		{
			model:      "anthropic.claude-v2",
			wantClaude: true,
			wantVision: false,
			wantTools:  false,
		},
		{
			model:      "amazon.titan-text-express-v1",
			wantTitan:  true,
			wantVision: false,
			wantTools:  false,
		},
		{
			model:      "meta.llama2-70b-chat-v1",
			wantLlama:  true,
			wantVision: false,
			wantTools:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			client := &Client{model: tt.model, provider: "bedrock"}

			if got := client.isClaudeModel(); got != tt.wantClaude {
				t.Errorf("isClaudeModel() = %v, want %v", got, tt.wantClaude)
			}
			if got := client.isTitanModel(); got != tt.wantTitan {
				t.Errorf("isTitanModel() = %v, want %v", got, tt.wantTitan)
			}
			if got := client.isLlamaModel(); got != tt.wantLlama {
				t.Errorf("isLlamaModel() = %v, want %v", got, tt.wantLlama)
			}

			// Check capabilities via GetModelInfo()
			modelInfo := client.GetModelInfo()
			if got := modelInfo.SupportsVision; got != tt.wantVision {
				t.Errorf("SupportsVision = %v, want %v", got, tt.wantVision)
			}
			if got := modelInfo.SupportsTools; got != tt.wantTools {
				t.Errorf("SupportsTools = %v, want %v", got, tt.wantTools)
			}
		})
	}
}
