// Package api 提供基于 Gin 的 HTTP 接口。
package api

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"godoc-rag/internal/rag"
)

// Handler 持有 RAG 实例。
type Handler struct {
	rag *rag.RAG
}

// New 构建 Handler。
func New(r *rag.RAG) *Handler { return &Handler{rag: r} }

type askRequest struct {
	Question string `json:"question" binding:"required"`
}

// Ask 处理 POST /ask，返回答案和引用来源。
func (h *Handler) Ask(c *gin.Context) {
	var req askRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "缺少必填字段 question"})
		return
	}

	ans, err := h.rag.Answer(c.Request.Context(), req.Question)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, ans)
}

// Health 健康检查。
func (h *Handler) Health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

// Router 组装路由。
func Router(h *Handler) *gin.Engine {
	r := gin.Default()
	r.GET("/health", h.Health)
	r.POST("/ask", h.Ask)
	return r
}
