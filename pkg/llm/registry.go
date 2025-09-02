// Provider registry to avoid import cycles
package llm

import (
	"sync"
)

// ProviderConstructor is a function that creates a new client for a provider
type ProviderConstructor func(config ClientConfig) (Client, error)

// providerRegistry holds all registered provider constructors
type providerRegistry struct {
	mu        sync.RWMutex
	providers map[string]ProviderConstructor
}

var registry = &providerRegistry{
	providers: make(map[string]ProviderConstructor),
}

// RegisterProvider registers a provider constructor function
func RegisterProvider(name string, constructor ProviderConstructor) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	registry.providers[name] = constructor
}

// GetProvider returns a provider constructor by name
func GetProvider(name string) (ProviderConstructor, bool) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	constructor, exists := registry.providers[name]
	return constructor, exists
}

// ListProviders returns all registered provider names
func ListProviders() []string {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	names := make([]string, 0, len(registry.providers))
	for name := range registry.providers {
		names = append(names, name)
	}
	return names
}
