// eval 是对 RAG 系统的评测工具：读取评测集，逐条跑完整链路，
// 再用 LLM-as-judge 从「忠实度」和「检索相关性」两个维度打分。
// 运行：go run ./cmd/eval -file data/eval/qa.jsonl
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"godoc-rag/internal/config"
	"godoc-rag/internal/llm"
	"godoc-rag/internal/rag"
	"godoc-rag/internal/store"
)

// QA 是评测集里的一条样本。
type QA struct {
	Question  string `json:"question"`
	Reference string `json:"reference"` // 参考答案，仅用于辅助 judge 判断
}

// JudgeResult 是 LLM-as-judge 给出的打分。
type JudgeResult struct {
	Faithful int `json:"faithful"` // 1=回答忠实于资料，0=存在编造
	Relevant int `json:"relevant"` // 1=检索到的资料相关，0=不相关
}

func main() {
	file := flag.String("file", "data/eval/qa.jsonl", "评测集文件（jsonl）")
	flag.Parse()

	cfg, err := config.Load()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	st, err := store.Open(ctx, cfg.DatabaseURL)
	if err != nil {
		log.Fatal(err)
	}
	defer st.Close()

	llmClient := llm.New(cfg.DashscopeAPIKey, cfg.DashscopeBaseURL)
	r := rag.New(llmClient, st, rag.Config{
		LLMModel: cfg.LLMModel, EmbedModel: cfg.EmbedModel, EmbedDim: cfg.EmbedDim,
		TopK: cfg.TopK, CandidateK: cfg.CandidateK, UseRerank: cfg.UseRerank,
	})

	qas, err := loadQA(*file)
	if err != nil {
		log.Fatal(err)
	}

	var sumFaith, sumRel float64
	n := 0
	for i, qa := range qas {
		ans, err := r.Answer(ctx, qa.Question)
		if err != nil {
			log.Printf("[%d] 链路错误: %v", i+1, err)
			continue
		}
		jr, err := judge(ctx, llmClient, cfg.LLMModel, qa, ans)
		if err != nil {
			log.Printf("[%d] 评测错误: %v", i+1, err)
			continue
		}
		n++
		sumFaith += float64(jr.Faithful)
		sumRel += float64(jr.Relevant)
		fmt.Printf("[%d] Q: %s\n    忠实度=%d 相关性=%d\n    A: %s\n\n",
			i+1, qa.Question, jr.Faithful, jr.Relevant, truncate(ans.Answer, 120))
	}

	if n == 0 {
		log.Fatal("没有成功评测任何一条样本，请先导入语料并检查评测集")
	}
	fmt.Printf("===== 汇总 =====\n样本数: %d\n忠实度: %.2f\n检索相关性: %.2f\n",
		n, sumFaith/float64(n), sumRel/float64(n))
}

// judge 用 LLM 从两个维度给单条结果打分，返回 JSON。
func judge(ctx context.Context, c *llm.Client, model string, qa QA, ans *rag.Answer) (JudgeResult, error) {
	var sb strings.Builder
	for _, s := range ans.Sources {
		sb.WriteString("- " + s.Content + "\n")
	}

	prompt := fmt.Sprintf(`你是评测专家。请判断这个 RAG 系统的回答质量。

问题：%s
参考答案：%s
检索到的资料：
%s
系统回答：%s

请输出一个 JSON：{"faithful": 0或1, "relevant": 0或1}
- faithful：回答是否忠实于资料（没有编造资料中不存在的内容），1=是 0=否
- relevant：检索到的资料是否与问题相关，1=相关 0=不相关
只输出 JSON，不要解释。`, qa.Question, qa.Reference, truncate(sb.String(), 2000), ans.Answer)

	resp, err := c.Chat(ctx, model, []llm.Message{{Role: "user", Content: prompt}})
	if err != nil {
		return JudgeResult{}, err
	}
	return parseJudge(resp)
}

// parseJudge 从 LLM 输出里截取 JSON 并解析。
func parseJudge(s string) (JudgeResult, error) {
	var jr JudgeResult
	if start := strings.Index(s, "{"); start >= 0 {
		if end := strings.LastIndex(s, "}"); end > start {
			s = s[start : end+1]
		}
	}
	if err := json.Unmarshal([]byte(s), &jr); err != nil {
		return JudgeResult{}, fmt.Errorf("解析 judge 输出失败: %w", err)
	}
	return jr, nil
}

func loadQA(path string) ([]QA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []QA
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var qa QA
		if err := json.Unmarshal([]byte(line), &qa); err != nil {
			return nil, fmt.Errorf("解析第 %d 行失败: %w", len(out)+1, err)
		}
		out = append(out, qa)
	}
	return out, sc.Err()
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "..."
}
