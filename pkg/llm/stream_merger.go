// Package llm provides abstractions for Large Language Model clients
// stream_merger.go defines utilities for merging multiple streams

package llm

import (
	"context"
	"sync"
)

// StreamMerger combines multiple streams into a single output stream
type StreamMerger struct {
	llmStream   <-chan StreamEvent
	toolStreams []<-chan StreamEvent
	output      chan StreamEvent
	ctx         context.Context
	cancel      context.CancelFunc
	wg          sync.WaitGroup
	once        sync.Once
}

// NewStreamMerger creates a new stream merger
func NewStreamMerger(ctx context.Context, llmStream <-chan StreamEvent, toolStreams []<-chan StreamEvent) *StreamMerger {
	ctx, cancel := context.WithCancel(ctx)
	return &StreamMerger{
		llmStream:   llmStream,
		toolStreams: toolStreams,
		output:      make(chan StreamEvent, 10),
		ctx:         ctx,
		cancel:      cancel,
	}
}

// Start begins merging streams and returns the output channel
func (sm *StreamMerger) Start() <-chan StreamEvent {
	sm.once.Do(func() {
		go sm.mergeStreams()
	})
	return sm.output
}

// Stop stops the stream merger
func (sm *StreamMerger) Stop() {
	sm.cancel()
	sm.wg.Wait()
}

func (sm *StreamMerger) mergeStreams() {
	defer close(sm.output)
	defer sm.cancel()

	// Start goroutines for LLM stream
	if sm.llmStream != nil {
		sm.wg.Add(1)
		go sm.forwardStream(sm.llmStream, "llm")
	}

	// Start goroutines for tool streams
	for i, toolStream := range sm.toolStreams {
		if toolStream != nil {
			sm.wg.Add(1)
			go sm.forwardStream(toolStream, "tool")
		}
		_ = i // avoid unused variable
	}

	// Wait for all streams to complete
	sm.wg.Wait()
}

func (sm *StreamMerger) forwardStream(stream <-chan StreamEvent, streamType string) {
	defer sm.wg.Done()

	for {
		select {
		case event, ok := <-stream:
			if !ok {
				return // Stream closed
			}

			// Forward the event with priority
			select {
			case sm.output <- event:
			case <-sm.ctx.Done():
				return
			}

		case <-sm.ctx.Done():
			return
		}
	}
}

// MergeStreams is a utility function to merge multiple streams
func MergeStreams(ctx context.Context, llmStream <-chan StreamEvent, toolStreams ...<-chan StreamEvent) <-chan StreamEvent {
	merger := NewStreamMerger(ctx, llmStream, toolStreams)
	return merger.Start()
}
