// server 是 RAG 服务 + Agent 的 HTTP 入口。
// 运行：go run ./cmd/server
package main

import (
	"context"
	"log"

	"godoc-rag/internal/agent"
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

	// 多知识库：列出当前所有命名空间，供 agent 工具描述提示用。
	namespaces, err := st.ListNamespaces(ctx)
	if err != nil {
		namespaces = []string{"go-docs"}
	}

	// ReAct Agent：让 LLM 能自主决定「直接回答」还是「调用工具」。
	tools := []agent.Tool{
		searchDocsTool(r, cfg.TopK, namespaces),
		countChunksTool(st, namespaces),
		currentTimeTool(),
	}
	if cfg.WebSearchAPIKey != "" {
		tools = append(tools, webSearchTool(cfg.WebSearchAPIKey))
	}

	a := agent.New(llmClient, tools, agent.Config{
		Model: cfg.LLMModel,
		SystemPrompt: "你是一个技术文档助手，背后有多个知识库。回答问题时，如果需要查资料就调用 search_docs 工具，" +
			"根据问题内容通过 namespace 参数选择正确的知识库；需要最新/实时信息就调用 web_search 联网搜索；" +
			"可以用 count_chunks 查知识库规模、current_time 查当前时间。已有信息足够时直接回答。" +
			"注意：拿到工具返回的结果后，请直接基于结果给出最终答案，不要反复调用同一个工具。",
	})

	router := api.Router(api.New(r, a))
	log.Printf("服务启动，监听 :%s", cfg.ServerPort)
	if err := router.Run(":" + cfg.ServerPort); err != nil {
		log.Fatalf("服务启动失败: %v", err)
	}
}
