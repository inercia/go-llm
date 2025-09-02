// Package registry provides provider registration and factory functionality for the go-llm framework.
package factory

import (
	"sync"

	"github.com/inercia/go-llm/pkg/llm"
)

// ProviderConstructor is a function that creates a new client for a provider
type ProviderConstructor func(config llm.ClientConfig) (llm.Client, error)

// providerRegistry holds all registered provider constructors
type providerRegistry struct {
	mu        sync.RWMutex
	providers map[string]ProviderConstructor
}

var globalRegistry = &providerRegistry{
	providers: make(map[string]ProviderConstructor),
}

// RegisterProvider registers a provider constructor function
func RegisterProvider(name string, constructor ProviderConstructor) {
	globalRegistry.mu.Lock()
	defer globalRegistry.mu.Unlock()
	globalRegistry.providers[name] = constructor
}

// GetProvider returns a provider constructor by name
func GetProvider(name string) (ProviderConstructor, bool) {
	globalRegistry.mu.RLock()
	defer globalRegistry.mu.RUnlock()
	constructor, exists := globalRegistry.providers[name]
	return constructor, exists
}

// ListProviders returns all registered provider names
func ListProviders() []string {
	globalRegistry.mu.RLock()
	defer globalRegistry.mu.RUnlock()

	names := make([]string, 0, len(globalRegistry.providers))
	for name := range globalRegistry.providers {
		names = append(names, name)
	}
	return names
}
