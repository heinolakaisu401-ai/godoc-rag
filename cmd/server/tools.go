package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"godoc-rag/internal/agent"
	"godoc-rag/internal/llm"
	"godoc-rag/internal/rag"
	"godoc-rag/internal/store"
)

// 下面这些是 agent 的内置工具。search_docs / count_chunks / current_time 不需要外部 key；
// web_search 需要博查 API key。

// nsHint 生成工具描述里关于「可选命名空间」的说明。
func nsHint(namespaces []string) string {
	return "可用的知识库命名空间：" + strings.Join(namespaces, "、") + "。不填默认 go-docs。"
}

// searchDocsTool 让 agent 能检索指定知识库。
func searchDocsTool(r *rag.RAG, topK int, namespaces []string) agent.Tool {
	hint := nsHint(namespaces)
	return agent.Tool{
		Def: llm.Tool{
			Type: "function",
			Function: llm.FunctionDef{
				Name:        "search_docs",
				Description: "检索指定知识库，返回与查询最相关的资料片段。需要查文档才能回答时调用，根据问题选择正确的 namespace。" + hint,
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query":     map[string]any{"type": "string", "description": "要检索的问题或关键词"},
						"namespace": map[string]any{"type": "string", "description": hint},
					},
					"required": []string{"query"},
				},
			},
		},
		Run: func(ctx context.Context, argsJSON string) (string, error) {
			var args struct {
				Query     string `json:"query"`
				Namespace string `json:"namespace"`
			}
			if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
				return "", fmt.Errorf("解析参数失败: %w", err)
			}
			ns := args.Namespace
			if ns == "" {
				ns = "go-docs"
			}
			docs, err := r.WithNamespace(ns).Search(ctx, args.Query, topK)
			if err != nil {
				return "", err
			}
			if len(docs) == 0 {
				return "没有找到相关资料。", nil
			}
			var sb strings.Builder
			for i, d := range docs {
				sb.WriteString(fmt.Sprintf("【资料%d】%s\n", i+1, truncate(d.Content, 500)))
			}
			return sb.String(), nil
		},
	}
}

// countChunksTool 让 agent 能查询指定知识库规模。
func countChunksTool(st *store.Store, namespaces []string) agent.Tool {
	hint := nsHint(namespaces)
	return agent.Tool{
		Def: llm.Tool{
			Type: "function",
			Function: llm.FunctionDef{
				Name:        "count_chunks",
				Description: "查询指定知识库有多少个文档片段（chunk）。" + hint,
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"namespace": map[string]any{"type": "string", "description": hint},
					},
				},
			},
		},
		Run: func(ctx context.Context, argsJSON string) (string, error) {
			var args struct {
				Namespace string `json:"namespace"`
			}
			if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
				return "", err
			}
			ns := args.Namespace
			if ns == "" {
				ns = "go-docs"
			}
			n, err := st.CountChunks(ctx, ns)
			if err != nil {
				return "", err
			}
			return fmt.Sprintf("知识库 %s 共有 %d 个 chunk。", ns, n), nil
		},
	}
}

// currentTimeTool 让 agent 知道当前时间。
func currentTimeTool() agent.Tool {
	return agent.Tool{
		Def: llm.Tool{
			Type: "function",
			Function: llm.FunctionDef{
				Name:        "current_time",
				Description: "返回当前时间。",
				Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
			},
		},
		Run: func(_ context.Context, _ string) (string, error) {
			return time.Now().Format(time.RFC3339), nil
		},
	}
}

// webSearchTool 让 agent 能联网搜索（需要博查 API key）。
func webSearchTool(apiKey string) agent.Tool {
	return agent.Tool{
		Def: llm.Tool{
			Type: "function",
			Function: llm.FunctionDef{
				Name:        "web_search",
				Description: "联网搜索，返回网页标题、链接和摘要。需要最新信息或文档库里没有的内容时调用。",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query": map[string]any{"type": "string", "description": "搜索关键词"},
					},
					"required": []string{"query"},
				},
			},
		},
		Run: func(ctx context.Context, argsJSON string) (string, error) {
			var args struct {
				Query string `json:"query"`
			}
			if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
				return "", fmt.Errorf("解析参数失败: %w", err)
			}
			return bochaSearch(ctx, apiKey, args.Query)
		},
	}
}

// bochaResponse 是博查返回的搜索结果：真正的 webPages 包在 data 字段里。
type bochaResponse struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
	Data struct {
		WebPages struct {
			Value []struct {
				Name    string `json:"name"`
				URL     string `json:"url"`
				Snippet string `json:"snippet"`
				Summary string `json:"summary"`
			} `json:"value"`
		} `json:"webPages"`
	} `json:"data"`
}

// bochaSearch 调用博查联网搜索，把前几条结果整理成纯文本喂回给 LLM。
func bochaSearch(ctx context.Context, apiKey, query string) (string, error) {
	payload, _ := json.Marshal(map[string]any{
		"query":     query,
		"freshness": "noLimit",
		"summary":   true,
		"count":     5,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://api.bochaai.com/v1/web-search", strings.NewReader(string(payload)))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		// 401/400/429 都会带原因（如 "Invalid API KEY"），直接透传给 LLM 看
		return "", fmt.Errorf("搜索接口返回 %d: %s", resp.StatusCode, truncate(string(raw), 200))
	}

	var out bochaResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", err
	}
	if out.Code != 200 {
		return "", fmt.Errorf("搜索接口错误(code=%d): %s", out.Code, out.Msg)
	}
	if len(out.Data.WebPages.Value) == 0 {
		return "没有搜到相关内容。", nil
	}

	var sb strings.Builder
	for i, p := range out.Data.WebPages.Value {
		text := p.Summary
		if text == "" {
			text = p.Snippet
		}
		sb.WriteString(fmt.Sprintf("【结果%d】%s\n链接：%s\n%s\n\n", i+1, p.Name, p.URL, truncate(text, 300)))
	}
	return sb.String(), nil
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "..."
}
