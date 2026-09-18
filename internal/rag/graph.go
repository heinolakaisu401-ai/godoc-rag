package rag

import (
	"context"
	"time"

	"godoc-rag/internal/graph"
	"godoc-rag/internal/store"
)

// 状态 key：图节点之间通过 State 流转的字段。
const (
	stateQuestion   = "question"
	stateQVec       = "qvec"      // 已向量化的问题；重试时复用，避免重复 embed
	stateContexts   = "contexts"
	stateAnswer     = "answer"
	stateRetrieveMs = "retrieve_ms"
	stateRerankMs   = "rerank_ms"
	stateGenerateMs = "generate_ms"
	stateRetries    = "retries" // 检索自纠已重试次数
	stateBroaden    = "broaden" // 本轮是否扩大召回
)

// 检索自纠的阈值：召回结果少于 minContextsToSkip 时，扩大召回重试，
// 最多重试 maxRetrieveRetries 次。
const (
	maxRetrieveRetries = 1
	minContextsToSkip  = 2
)

// buildGraph 把 RAG 链路拆成一张状态图，节点与现有 retrieve/rerank/generate 一一对应：
//
//	retrieve ─▶ rerank ─▶ route ─┬─▶ generate ─▶ END
//	                    │        └─▶ no_context ─▶ END
//	                    └──────────▶ retrieve（召回太少，扩大后重试）
//
// route 是条件边（nextAfterRerank）：召回结果太少且未重试过，就扩大召回回到
// retrieve 再来一次；重试后仍为空才走 no_context 优雅兜底，不浪费 LLM 调用。
func (r *RAG) buildGraph(onDelta func(string) error) *graph.Graph {
	g := graph.New()

	g.AddNode("retrieve", func(ctx context.Context, s graph.State) error {
		q := s[stateQuestion].(string)
		qvec, ok := s[stateQVec].([]float32)
		if !ok { // 第一次：向量化并缓存；重试时直接复用
			var err error
			qvec, err = r.llm.Embed(ctx, r.cfg.EmbedModel, q, r.cfg.EmbedDim)
			if err != nil {
				return err
			}
			s[stateQVec] = qvec
		}
		n := r.cfg.CandidateK
		if b, _ := s[stateBroaden].(bool); b {
			n *= 2 // 扩大召回
		}
		t0 := time.Now()
		cands, err := r.retrieve(ctx, q, qvec, n)
		if err != nil {
			return err
		}
		s[stateContexts] = cands
		s[stateRetrieveMs] = msSince(t0)
		return nil
	})

	g.AddNode("rerank", func(ctx context.Context, s graph.State) error {
		q := s[stateQuestion].(string)
		cands := s[stateContexts].([]store.Retrieved)
		t0 := time.Now()
		if r.cfg.UseRerank && len(cands) > r.cfg.TopK {
			out, err := r.rerank(ctx, q, cands, r.cfg.TopK)
			if err != nil {
				return err
			}
			cands = out
		} else if len(cands) > r.cfg.TopK {
			cands = cands[:r.cfg.TopK]
		}
		s[stateContexts] = cands
		s[stateRerankMs] = msSince(t0)
		return nil
	})

	g.AddNode("generate", func(ctx context.Context, s graph.State) error {
		q := s[stateQuestion].(string)
		cands := s[stateContexts].([]store.Retrieved)
		t0 := time.Now()
		text, err := r.generate(ctx, q, cands, onDelta)
		if err != nil {
			return err
		}
		s[stateAnswer] = text
		s[stateGenerateMs] = msSince(t0)
		return nil
	})

	g.AddNode("no_context", func(_ context.Context, s graph.State) error {
		s[stateAnswer] = "根据现有资料无法回答该问题。"
		s[stateGenerateMs] = 0.0
		return nil
	})

	g.SetEntryPoint("retrieve")
	g.AddEdge("retrieve", "rerank")
	g.AddConditionalEdges("rerank", nextAfterRerank)
	g.AddEdge("generate", graph.End)
	g.AddEdge("no_context", graph.End)

	return g
}

// nextAfterRerank 是 rerank 之后的条件路由：决定下一个节点，并负责重试计数与扩大召回。
//   - 召回结果太少（< minContextsToSkip）且未到重试上限：扩大召回，回到 retrieve；
//   - 重试后仍无结果：走 no_context 优雅兜底；
//   - 否则：走 generate。
func nextAfterRerank(s graph.State) string {
	contexts, _ := s[stateContexts].([]store.Retrieved)
	retries, _ := s[stateRetries].(int)
	if len(contexts) < minContextsToSkip && retries < maxRetrieveRetries {
		s[stateRetries] = retries + 1
		s[stateBroaden] = true
		return "retrieve"
	}
	if len(contexts) == 0 {
		return "no_context"
	}
	return "generate"
}
