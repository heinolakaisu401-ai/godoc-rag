// Package api 提供基于 Gin 的 HTTP 接口。
package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"

	"godoc-rag/internal/agent"
	"godoc-rag/internal/llm"
	"godoc-rag/internal/rag"
)

// Handler 持有 RAG 实例与 Agent 实例。
type Handler struct {
	rag      *rag.RAG
	agent    *agent.Agent
	sessions *sessionStore
}

// New 构建 Handler。
func New(r *rag.RAG, a *agent.Agent) *Handler {
	return &Handler{rag: r, agent: a, sessions: newSessionStore()}
}

type askRequest struct {
	Question  string `json:"question" binding:"required"`
	Namespace string `json:"namespace"` // 可选，默认 go-docs
	Stream    bool   `json:"stream"`    // true 时用 SSE 流式返回（打字机效果）
}

type agentRequest struct {
	Question  string `json:"question" binding:"required"`
	SessionID string `json:"session_id"` // 可选：传了就启用多轮记忆
	Stream    bool   `json:"stream"`     // true 时用 SSE 流式返回（打字机效果）
}

// Ask 处理 POST /ask，返回答案和引用来源（纯 RAG）。stream=true 时以 SSE 逐字输出。
func (h *Handler) Ask(c *gin.Context) {
	var req askRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "缺少必填字段 question"})
		return
	}

	ns := req.Namespace
	if ns == "" {
		ns = "go-docs"
	}
	if !req.Stream {
		ans, err := h.rag.WithNamespace(ns).Answer(c.Request.Context(), req.Question)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, ans)
		return
	}

	flusher := sseWriter(c)
	ans, err := h.rag.WithNamespace(ns).AnswerStream(c.Request.Context(), req.Question, func(delta string) error {
		return writeSSE(c.Writer, flusher, gin.H{"type": "delta", "content": delta})
	})
	if err != nil {
		_ = writeSSE(c.Writer, flusher, gin.H{"type": "error", "message": err.Error()})
		return
	}
	_ = writeSSE(c.Writer, flusher, gin.H{
		"type":    "done",
		"answer":  ans.Answer,
		"sources": ans.Sources,
		"timing":  ans.Timing,
	})
}

// Agent 处理 POST /agent，跑 ReAct agent（工具调用 + 多轮记忆）。stream=true 时 SSE 逐字输出。
func (h *Handler) Agent(c *gin.Context) {
	var req agentRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "缺少必填字段 question"})
		return
	}

	history := h.sessions.get(req.SessionID)
	if !req.Stream {
		ans, err := h.agent.RunWithHistory(c.Request.Context(), history, req.Question)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		h.remember(req.SessionID, req.Question, ans)
		c.JSON(http.StatusOK, gin.H{"answer": ans})
		return
	}

	flusher := sseWriter(c)
	ans, err := h.agent.RunWithHistoryStream(c.Request.Context(), history, req.Question, func(delta string) error {
		return writeSSE(c.Writer, flusher, gin.H{"type": "delta", "content": delta})
	})
	if err != nil {
		_ = writeSSE(c.Writer, flusher, gin.H{"type": "error", "message": err.Error()})
		return
	}
	h.remember(req.SessionID, req.Question, ans)
	_ = writeSSE(c.Writer, flusher, gin.H{"type": "done", "answer": ans})
}

// remember 把一轮问答写进会话记忆（session_id 为空则不记）。
func (h *Handler) remember(sessionID, question, answer string) {
	if sessionID == "" {
		return
	}
	h.sessions.append(sessionID,
		llm.Message{Role: "user", Content: question},
		llm.Message{Role: "assistant", Content: answer},
	)
}

// Health 健康检查。
func (h *Handler) Health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

// sseWriter 设置 SSE 响应头并返回用于刷新缓冲的 Flusher。
func sseWriter(c *gin.Context) http.Flusher {
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("X-Accel-Buffering", "no") // 关掉 nginx 等反代的缓冲
	c.Writer.WriteHeader(http.StatusOK)
	f, _ := c.Writer.(http.Flusher)
	return f
}

// writeSSE 以 SSE 的 data 行格式输出一个 JSON 事件并立即刷新到客户端。
func writeSSE(w http.ResponseWriter, f http.Flusher, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
		return err
	}
	if f != nil {
		f.Flush()
	}
	return nil
}

// Router 组装路由。
func Router(h *Handler) *gin.Engine {
	r := gin.Default()
	r.GET("/health", h.Health)
	r.POST("/ask", h.Ask)
	r.POST("/agent", h.Agent)
	r.StaticFile("/", "web/index.html") // 演示前端（浏览器打开 http://localhost:8080/）
	return r
}

// sessionStore 是内存版会话记忆：按 session_id 保存最近若干轮对话。
// 重启即清空；要持久化可换成存 PostgreSQL（见「多轮记忆持久化」清单）。
type sessionStore struct {
	mu sync.Mutex
	m  map[string][]llm.Message
}

func newSessionStore() *sessionStore {
	return &sessionStore{m: map[string][]llm.Message{}}
}

func (s *sessionStore) get(id string) []llm.Message {
	if id == "" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]llm.Message(nil), s.m[id]...)
}

func (s *sessionStore) append(id string, msgs ...llm.Message) {
	if id == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m[id] = append(s.m[id], msgs...)
	// 只保留最近 20 条，避免内存无限增长。
	if len(s.m[id]) > 20 {
		s.m[id] = s.m[id][len(s.m[id])-20:]
	}
}
