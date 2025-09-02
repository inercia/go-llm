package factory

import (
	"github.com/inercia/go-llm/pkg/llm"
	"github.com/inercia/go-llm/pkg/providers/deepseek"
	"github.com/inercia/go-llm/pkg/providers/gemini"
	"github.com/inercia/go-llm/pkg/providers/mock"
	"github.com/inercia/go-llm/pkg/providers/ollama"
	"github.com/inercia/go-llm/pkg/providers/openai"
	"github.com/inercia/go-llm/pkg/providers/openrouter"
)

func init() {
	// Register the openrouter provider
	RegisterProvider("openrouter", func(config llm.ClientConfig) (llm.Client, error) {
		return openrouter.NewClient(config)
	})

	// Register the OpenAI provider
	RegisterProvider("openai", func(config llm.ClientConfig) (llm.Client, error) {
		return openai.NewClient(config)
	})

	// Register the deepseek provider
	RegisterProvider("deepseek", func(config llm.ClientConfig) (llm.Client, error) {
		return deepseek.NewClient(config)
	})

	// Register the gemini provider
	RegisterProvider("gemini", func(config llm.ClientConfig) (llm.Client, error) {
		return gemini.NewClient(config)
	})

	// Register the ollama provider
	RegisterProvider("ollama", func(config llm.ClientConfig) (llm.Client, error) {
		return ollama.NewClient(config)
	})

	// Register the mock provider
	RegisterProvider("mock", func(config llm.ClientConfig) (llm.Client, error) {
		return mock.NewClient(config.Model, "mock")
	})
	RegisterProvider("mocked", func(config llm.ClientConfig) (llm.Client, error) {
		return mock.NewClient(config.Model, "mock")
	})
}
