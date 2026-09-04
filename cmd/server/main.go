// server 是 RAG 服务的 HTTP 入口。
// 运行：go run ./cmd/server
package main

import (
	"context"
	"log"

	"godoc-rag/internal/api"
	"godoc-rag/internal/config"
	"godoc-rag/internal/llm"
	"godoc-rag/internal/rag"
	"godoc-rag/internal/store"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("加载配置失败: %v", err)
	}
	if cfg.DashscopeAPIKey == "" {
		log.Fatal("缺少 DASHSCOPE_API_KEY，请先复制 .env.example 为 .env 并填入 key")
	}

	ctx := context.Background()

	st, err := store.Open(ctx, cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("连接数据库失败: %v", err)
	}
	defer st.Close()

	llmClient := llm.New(cfg.DashscopeAPIKey, cfg.DashscopeBaseURL)
	r := rag.New(llmClient, st, rag.Config{
		LLMModel:   cfg.LLMModel,
		EmbedModel: cfg.EmbedModel,
		EmbedDim:   cfg.EmbedDim,
		TopK:       cfg.TopK,       // 最终给 LLM 的上下文块数
		CandidateK: cfg.CandidateK, // 检索候选数（先多召回，再重排精选）
		UseRerank:  cfg.UseRerank,
	})

	router := api.Router(api.New(r))
	log.Printf("服务启动，监听 :%s", cfg.ServerPort)
	if err := router.Run(":" + cfg.ServerPort); err != nil {
		log.Fatalf("服务启动失败: %v", err)
	}
}
