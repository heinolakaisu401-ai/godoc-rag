// Package rag 编排完整的检索增强生成（RAG）链路：
// embed 问题 → 混合检索（向量 + 关键词）→ 重排 → 拼 prompt 生成答案。
package rag

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"godoc-rag/internal/chunker"
	"godoc-rag/internal/llm"
	"godoc-rag/internal/store"
)

// Config 是 RAG 链路的运行参数。
type Config struct {
	LLMModel   string
	EmbedModel string
	EmbedDim   int
	TopK       int  // 最终喂给 LLM 的上下文块数
	CandidateK int  // 检索阶段返回的候选数（先多召回，再重排精选）
	UseRerank  bool // 是否启用 LLM 重排
}

// RAG 是核心结构。
type RAG struct {
	llm   *llm.Client
	store *store.Store
	cfg   Config
}

// New 构建 RAG 实例。
func New(llmClient *llm.Client, st *store.Store, cfg Config) *RAG {
	return &RAG{llm: llmClient, store: st, cfg: cfg}
}

// Answer 是最终返回给用户的结果。
type Answer struct {
	Question string   `json:"question"`
	Answer   string   `json:"answer"`
	Sources  []Source `json:"sources"` // 引用的来源，便于核查是否有幻觉
	Timing   Timing   `json:"timing"`  // 各阶段耗时（毫秒），用于评测和性能观测
}

// Timing 记录一次问答链路各阶段的耗时，单位毫秒。
type Timing struct {
	RetrieveMs float64 `json:"retrieve_ms"` // 检索耗时（向量检索 + 关键词检索 + RRF 融合）
	RerankMs   float64 `json:"rerank_ms"`   // LLM 重排耗时（未启用重排时为 0）
	GenerateMs float64 `json:"generate_ms"` // LLM 生成答案耗时
	TotalMs    float64 `json:"total_ms"`    // 端到端总耗时（不含评测 judge 调用）
}

// Source 是引用的一个资料片段。
type Source struct {
	Content    string  `json:"content"`
	Section    string  `json:"section"`
	Source     string  `json:"source"`
	Score      float64 `json:"score"`
	CodeStatus string  `json:"code_status,omitempty"` // 代码块完整性:none/complete/missing_end/missing_start
}

// Answer 走完整链路回答问题。
func (r *RAG) Answer(ctx context.Context, question string) (*Answer, error) {
	start := time.Now()

	qvec, err := r.llm.Embed(ctx, r.cfg.EmbedModel, question, r.cfg.EmbedDim)
	if err != nil {
		return nil, fmt.Errorf("向量化问题失败: %w", err)
	}

	retrieveStart := time.Now()
	cands, err := r.retrieve(ctx, question, qvec, r.cfg.CandidateK)
	retrieveMs := msSince(retrieveStart)
	if err != nil {
		return nil, err
	}

	rerankMs := 0.0
	if r.cfg.UseRerank && len(cands) > r.cfg.TopK {
		rerankStart := time.Now()
		cands, err = r.rerank(ctx, question, cands, r.cfg.TopK)
		rerankMs = msSince(rerankStart)
		if err != nil {
			return nil, err
		}
	} else if len(cands) > r.cfg.TopK {
		cands = cands[:r.cfg.TopK]
	}

	generateStart := time.Now()
	text, err := r.generate(ctx, question, cands)
	generateMs := msSince(generateStart)
	if err != nil {
		return nil, err
	}

	sources := make([]Source, 0, len(cands))
	for _, c := range cands {
		sources = append(sources, Source{
			Content: c.Content, Section: c.Section, Source: c.Source, Score: c.Score,
			CodeStatus: codeStatusOf(c.Content),
		})
	}
	return &Answer{
		Question: question,
		Answer:   text,
		Sources:  sources,
		Timing: Timing{
			RetrieveMs: retrieveMs,
			RerankMs:   rerankMs,
			GenerateMs: generateMs,
			TotalMs:    msSince(start),
		},
	}, nil
}

// retrieve 做混合检索：向量检索 + 关键词检索，用 RRF（倒数排名融合）合并。
func (r *RAG) retrieve(ctx context.Context, q string, qvec []float32, n int) ([]store.Retrieved, error) {
	vecRes, err := r.store.VectorSearch(ctx, qvec, n)
	if err != nil {
		return nil, err
	}
	kwRes, err := r.store.KeywordSearch(ctx, q, n)
	if err != nil {
		return nil, err
	}
	return rrf(vecRes, kwRes, n), nil
}

// rerank 用一次 LLM 调用对候选片段重排，只保留最相关的 TopK。
func (r *RAG) rerank(ctx context.Context, q string, cands []store.Retrieved, k int) ([]store.Retrieved, error) {
	var sb strings.Builder
	sb.WriteString("你是检索相关性排序器。下面是问题，以及若干候选文档片段。\n\n")
	sb.WriteString("问题：" + q + "\n\n候选片段：\n")
	for i, c := range cands {
		sb.WriteString(fmt.Sprintf("[%d] %s\n", i+1, truncate(c.Content, 200)))
	}
	sb.WriteString("\n请只输出与问题最相关的片段编号，按相关性从高到低排列，用逗号分隔。只输出数字，不要解释。\n")

	resp, err := r.llm.Chat(ctx, r.cfg.LLMModel, []llm.Message{{Role: "user", Content: sb.String()}})
	if err != nil {
		return nil, err
	}

	out := make([]store.Retrieved, 0, k)
	seen := map[int64]bool{}
	for _, idx := range parseIndices(resp, len(cands)) { // idx 已是 0 基
		if !seen[cands[idx].ID] {
			out = append(out, cands[idx])
			seen[cands[idx].ID] = true
			if len(out) >= k {
				break
			}
		}
	}
	// 兜底：如果解析失败或不够 k 个，按原顺序补足。
	for _, c := range cands {
		if len(out) >= k {
			break
		}
		if !seen[c.ID] {
			out = append(out, c)
			seen[c.ID] = true
		}
	}
	return out, nil
}

// generate 把上下文和问题拼进 prompt，交给 LLM 生成答案。
func (r *RAG) generate(ctx context.Context, q string, ctxChunks []store.Retrieved) (string, error) {
	var sb strings.Builder
	sb.WriteString("你是一个 Go 语言文档助手。请只根据下面提供的资料回答用户问题。\n")
	sb.WriteString("如果资料中找不到答案，请明确说「根据现有资料无法回答」，不要编造。\n\n")
	sb.WriteString("【代码块完整性规则】检索资料中的代码块可能因切分而残缺：\n")
	sb.WriteString("- 标注「不完整（缺结尾）」的，只依据已给出的部分作答，禁止补全、续写或猜测缺失部分；不要丢弃该段。\n")
	sb.WriteString("- 标注「不完整（缺开头）」的，只是某段代码的结尾，禁止据此推断完整代码。\n")
	sb.WriteString("- 若问题必须依赖缺失部分才能回答，请说「该段资料不完整，无法据此给出完整答案」。\n\n")
	sb.WriteString("资料：\n")
	for i, c := range ctxChunks {
		sb.WriteString(fmt.Sprintf("[%d] %s\n", i+1, formatChunk(c.Content)))
	}
	sb.WriteString("\n问题：" + q + "\n")

	return r.llm.Chat(ctx, r.cfg.LLMModel, []llm.Message{
		{Role: "system", Content: "你是严谨、只依据给定资料作答的 Go 文档助手。"},
		{Role: "user", Content: sb.String()},
	})
}

// ---------- 工具函数 ----------

const rrfK = 60.0

// rrf 用倒数排名融合合并两路检索结果，避免被单路结果主导。
func rrf(a, b []store.Retrieved, limit int) []store.Retrieved {
	scores := map[int64]float64{}
	docs := map[int64]store.Retrieved{}
	add := func(list []store.Retrieved) {
		for i, d := range list {
			scores[d.ID] += 1.0 / (rrfK + float64(i+1))
			docs[d.ID] = d
		}
	}
	add(a)
	add(b)

	ids := make([]int64, 0, len(scores))
	for id := range scores {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return scores[ids[i]] > scores[ids[j]] })

	out := make([]store.Retrieved, 0, limit)
	for _, id := range ids {
		d := docs[id]
		d.Score = scores[id]
		out = append(out, d)
		if len(out) >= limit {
			break
		}
	}
	return out
}

// parseIndices 从 LLM 输出里解析出 0 基下标（提示词里用的是 1 基编号）。
func parseIndices(s string, max int) []int {
	var out []int
	for _, part := range strings.FieldsFunc(s, func(r rune) bool { return r < '0' || r > '9' }) {
		var n int
		if _, err := fmt.Sscanf(part, "%d", &n); err == nil && n >= 1 && n <= max {
			out = append(out, n-1)
		}
	}
	return out
}

// msSince 返回从 t 到现在的耗时，单位毫秒。
func msSince(t time.Time) float64 {
	return float64(time.Since(t).Microseconds()) / 1000.0
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "..."
}

// formatChunk 根据代码块完整性给 chunk 内容加提示前缀，提示 LLM 不要补全/推断残缺代码。
func formatChunk(content string) string {
	switch chunker.Summarize(chunker.CheckCodeBlocks(content)) {
	case chunker.DanglingOpen:
		return "（代码不完整：缺结尾）以下代码块缺少结尾，请仅使用已给出的上半部分，严禁补全：\n" + content
	case chunker.DanglingClose:
		return "（代码不完整：缺开头）以下仅为代码结尾，开头未包含，禁止据此推断完整代码：\n" + content
	default:
		return content
	}
}

// codeStatusOf 把代码块完整性状态映射为 JSON 友好的字符串，写入返回的 sources 供核查。
func codeStatusOf(content string) string {
	switch chunker.Summarize(chunker.CheckCodeBlocks(content)) {
	case chunker.DanglingOpen:
		return "missing_end"
	case chunker.DanglingClose:
		return "missing_start"
	case chunker.Complete:
		return "complete"
	default:
		return "none"
	}
}
