package agent

import (
	"context"
	"errors"
	"testing"

	"godoc-rag/internal/llm"
)

// fakeChatter 按顺序返回预设的 assistant 消息，用于测试 ReAct 循环，无需真实 LLM。
type fakeChatter struct {
	responses []llm.Message
	idx       int
}

func (f *fakeChatter) ChatWithTools(_ context.Context, _ string, _ []llm.Message, _ []llm.Tool) (llm.Message, error) {
	if f.idx >= len(f.responses) {
		return llm.Message{}, errors.New("没有更多预设响应")
	}
	m := f.responses[f.idx]
	f.idx++
	return m, nil
}

// TestAgentToolLoop 验证：第一轮要求调工具 → 执行工具 → 把结果喂回 → 第二轮给出最终答案。
func TestAgentToolLoop(t *testing.T) {
	fake := &fakeChatter{responses: []llm.Message{
		{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{{
				ID:       "call_1",
				Type:     "function",
				Function: llm.ToolCallFunction{Name: "echo", Arguments: `{"x":"hi"}`},
			}},
		},
		{Role: "assistant", Content: "最终答案"},
	}}

	echo := Tool{
		Def: llm.Tool{Type: "function", Function: llm.FunctionDef{Name: "echo", Description: "回显", Parameters: map[string]any{"type": "object"}}},
		Run: func(_ context.Context, argsJSON string) (string, error) { return argsJSON, nil },
	}

	a := New(fake, []Tool{echo}, Config{Model: "test-model"})
	got, err := a.Run(context.Background(), "问题")
	if err != nil {
		t.Fatalf("Run 失败: %v", err)
	}
	if got != "最终答案" {
		t.Fatalf("got %q, want 最终答案", got)
	}
	if fake.idx != 2 {
		t.Fatalf("应调用 2 次 LLM，实际 %d 次", fake.idx)
	}
}

// TestAgentToolNotFound 验证：模型调用了未注册的工具时，循环不会崩，LLM 能据此继续。
func TestAgentToolNotFound(t *testing.T) {
	fake := &fakeChatter{responses: []llm.Message{
		{Role: "assistant", ToolCalls: []llm.ToolCall{{ID: "c1", Type: "function", Function: llm.ToolCallFunction{Name: "nope", Arguments: "{}"}}}},
		{Role: "assistant", Content: "done"},
	}}
	a := New(fake, nil, Config{Model: "m"})
	got, err := a.Run(context.Background(), "q")
	if err != nil {
		t.Fatalf("未知工具不应导致 Run 失败，got %v", err)
	}
	if got != "done" {
		t.Fatalf("got %q, want done", got)
	}
}
