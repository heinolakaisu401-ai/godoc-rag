// Package agent 实现一个 ReAct 风格的 agent：
// LLM 自主决定「直接回答」还是「调用工具」，调用后把结果喂回去继续决策，
// 直到给出最终答案。这个「决策 → 执行 → 观察」的循环由 internal/graph 的状态图引擎驱动。
package agent

import (
	"context"
	"fmt"
	"log"

	"godoc-rag/internal/graph"
	"godoc-rag/internal/llm"
)

// ChatCompleter 抽象「带工具调用的对话补全」，便于在测试里注入假实现。
type ChatCompleter interface {
	ChatWithTools(ctx context.Context, model string, messages []llm.Message, tools []llm.Tool) (llm.Message, error)
}

// StreamChatter 抽象「带工具调用的流式对话补全」，用于逐字输出最终答案。
// llm.Client 实现了它；测试里的假实现只实现 ChatCompleter 即可，此时自动退回非流式。
type StreamChatter interface {
	StreamChat(ctx context.Context, model string, messages []llm.Message, tools []llm.Tool, onDelta func(string) error) (llm.Message, error)
}

// Tool 是 agent 可调用的一个工具：Def 给 LLM 看的描述，Run 实际执行。
type Tool struct {
	Def llm.Tool
	Run func(ctx context.Context, argsJSON string) (string, error)
}

// Config 是 agent 的运行参数。
type Config struct {
	Model        string
	SystemPrompt string
	MaxIter      int // 最多几轮 LLM 调用（工具调用轮数上限），防止无限循环
}

// Agent 是 ReAct 循环的 agent。
type Agent struct {
	llm   ChatCompleter
	tools []Tool
	cfg   Config
}

// New 构建 agent。
func New(c ChatCompleter, tools []Tool, cfg Config) *Agent {
	if cfg.MaxIter <= 0 {
		cfg.MaxIter = 6
	}
	if cfg.SystemPrompt == "" {
		cfg.SystemPrompt = "你是一个能调用工具的助手。需要查资料时调用工具，否则直接回答。"
	}
	return &Agent{llm: c, tools: tools, cfg: cfg}
}

// Run 跑一轮 agent 循环（无历史），返回最终答案。
func (a *Agent) Run(ctx context.Context, question string) (string, error) {
	return a.run(ctx, nil, question, nil)
}

// RunWithHistory 带历史消息跑 agent（用于多轮记忆）。
func (a *Agent) RunWithHistory(ctx context.Context, history []llm.Message, question string) (string, error) {
	return a.run(ctx, history, question, nil)
}

// RunStream 与 Run 相同，但最终答案以流式逐字回调 onDelta。
func (a *Agent) RunStream(ctx context.Context, question string, onDelta func(string) error) (string, error) {
	return a.run(ctx, nil, question, onDelta)
}

// RunWithHistoryStream 带历史消息流式跑 agent。
func (a *Agent) RunWithHistoryStream(ctx context.Context, history []llm.Message, question string, onDelta func(string) error) (string, error) {
	return a.run(ctx, history, question, onDelta)
}

// run 是 agent 循环的公共实现：onDelta 为 nil 时走非流式，否则尽量流式输出最终答案。
func (a *Agent) run(ctx context.Context, history []llm.Message, question string, onDelta func(string) error) (string, error) {
	messages := make([]llm.Message, 0, len(history)+2)
	messages = append(messages, llm.Message{Role: "system", Content: a.cfg.SystemPrompt})
	messages = append(messages, history...)
	messages = append(messages, llm.Message{Role: "user", Content: question})

	g := graph.New()
	g.SetMaxSteps(a.cfg.MaxIter*2 + 4)

	g.AddNode("agent", func(ctx context.Context, s graph.State) error {
		iter, _ := s["iter"].(int)
		if iter >= a.cfg.MaxIter {
			return fmt.Errorf("达到最大工具调用轮数 %d，仍未给出最终答案", a.cfg.MaxIter)
		}
		s["iter"] = iter + 1

		msgs := s["messages"].([]llm.Message)
		resp, err := a.call(ctx, msgs, onDelta)
		if err != nil {
			return err
		}
		msgs = append(msgs, resp)
		s["messages"] = msgs
		if len(resp.ToolCalls) == 0 {
			s["final"] = resp.Content // 没有工具调用 = 最终答案
			log.Printf("[agent] 第 %d 轮：给出最终答案", iter+1)
		} else {
			for _, tc := range resp.ToolCalls {
				log.Printf("[agent] 第 %d 轮：调用工具 %s(%s)", iter+1, tc.Function.Name, tc.Function.Arguments)
			}
		}
		return nil
	})

	g.AddNode("tools", func(ctx context.Context, s graph.State) error {
		msgs := s["messages"].([]llm.Message)
		last := msgs[len(msgs)-1]
		for _, tc := range last.ToolCalls {
			result := runTool(ctx, a.tools, tc.Function.Name, tc.Function.Arguments)
			log.Printf("[agent] 工具 %s 返回 %d 字符", tc.Function.Name, len(result))
			msgs = append(msgs, llm.Message{
				Role:       "tool",
				ToolCallID: tc.ID,
				Name:       tc.Function.Name,
				Content:    result,
			})
		}
		s["messages"] = msgs
		return nil
	})

	g.SetEntryPoint("agent")
	g.AddEdge("tools", "agent") // 工具结果喂回去，形成 ReAct 循环
	g.AddConditionalEdges("agent", func(s graph.State) string {
		if _, ok := s["final"].(string); ok {
			return graph.End
		}
		return "tools"
	})

	state, err := g.Run(ctx, graph.State{"messages": messages})
	if err != nil {
		return "", err
	}
	final, _ := state["final"].(string)
	if final == "" {
		return "", fmt.Errorf("agent 未产出最终答案")
	}
	return final, nil
}

// call 决定用流式还是普通方式调用模型：仅当需要流式输出且底层支持流式时才走流式。
func (a *Agent) call(ctx context.Context, msgs []llm.Message, onDelta func(string) error) (llm.Message, error) {
	if onDelta != nil {
		if streamer, ok := a.llm.(StreamChatter); ok {
			return streamer.StreamChat(ctx, a.cfg.Model, msgs, toolDefs(a.tools), onDelta)
		}
	}
	return a.llm.ChatWithTools(ctx, a.cfg.Model, msgs, toolDefs(a.tools))
}

// toolDefs 把 Tool 列表转成给 LLM 的工具定义。
func toolDefs(tools []Tool) []llm.Tool {
	out := make([]llm.Tool, 0, len(tools))
	for _, t := range tools {
		out = append(out, t.Def)
	}
	return out
}

// runTool 按名字找到工具并执行，返回结果文本。
// 出错或遇到未知工具也返回文本（而非 error），让 LLM 能据此自我修正。
func runTool(ctx context.Context, tools []Tool, name, argsJSON string) string {
	for _, t := range tools {
		if t.Def.Function.Name == name {
			res, err := t.Run(ctx, argsJSON)
			if err != nil {
				return "工具调用出错: " + err.Error()
			}
			return res
		}
	}
	return "未知工具: " + name
}
