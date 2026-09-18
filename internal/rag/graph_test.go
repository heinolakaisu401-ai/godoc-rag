package rag

import (
	"testing"

	"godoc-rag/internal/graph"
	"godoc-rag/internal/store"
)

// TestNextAfterRerankRetriesOnWeakRecall 召回为空且未重试过：应扩大召回回到 retrieve。
func TestNextAfterRerankRetriesOnWeakRecall(t *testing.T) {
	s := graph.State{stateContexts: []store.Retrieved{}}
	if got := nextAfterRerank(s); got != "retrieve" {
		t.Fatalf("空召回首次应重试，got %q", got)
	}
	if got := s[stateRetries].(int); got != 1 {
		t.Fatalf("重试计数应为 1，got %v", got)
	}
	if got := s[stateBroaden].(bool); got != true {
		t.Fatalf("应标记为扩大召回")
	}
}

// TestNextAfterRerankFallsBackWhenRetriesExhausted 重试后仍空：走 no_context 兜底。
func TestNextAfterRerankFallsBackWhenRetriesExhausted(t *testing.T) {
	s := graph.State{stateContexts: []store.Retrieved{}, stateRetries: 1}
	if got := nextAfterRerank(s); got != "no_context" {
		t.Fatalf("重试后仍空应兜底，got %q", got)
	}
}

// TestNextAfterRerankGeneratesWhenEnough 召回足够（达到阈值）：正常生成。
func TestNextAfterRerankGeneratesWhenEnough(t *testing.T) {
	s := graph.State{stateContexts: []store.Retrieved{{ID: 1}, {ID: 2}}}
	if got := nextAfterRerank(s); got != "generate" {
		t.Fatalf("召回足够应生成，got %q", got)
	}
}
