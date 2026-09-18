package graph

import (
	"context"
	"testing"
)

// TestRunBranchAndLoop 验证图引擎支持「条件分支 + 循环」。
func TestRunBranchAndLoop(t *testing.T) {
	g := New()
	g.SetEntryPoint("start")

	g.AddNode("start", func(_ context.Context, s State) error {
		s["n"] = 0
		return nil
	})
	g.AddNode("inc", func(_ context.Context, s State) error {
		s["n"] = s["n"].(int) + 1
		return nil
	})
	g.AddNode("done", func(_ context.Context, s State) error {
		s["answer"] = "ok"
		return nil
	})

	g.AddEdge("start", "inc")
	// 条件边：n 满 3 次就结束，否则循环回 inc 自己。
	g.AddConditionalEdges("inc", func(s State) string {
		if s["n"].(int) >= 3 {
			return "done"
		}
		return "inc"
	})
	g.AddEdge("done", End)

	state, err := g.Run(context.Background(), State{})
	if err != nil {
		t.Fatalf("Run 失败: %v", err)
	}
	if got := state["n"].(int); got != 3 {
		t.Fatalf("n = %d, want 3", got)
	}
	if got := state["answer"]; got != "ok" {
		t.Fatalf("answer = %v, want ok", got)
	}
}

// TestRunMaxSteps 验证循环保护：死循环会在步数超限时中止。
func TestRunMaxSteps(t *testing.T) {
	g := New()
	g.SetMaxSteps(5)
	g.SetEntryPoint("loop")
	g.AddNode("loop", func(_ context.Context, s State) error { return nil })
	g.AddEdge("loop", "loop") // 永远循环

	if _, err := g.Run(context.Background(), State{}); err == nil {
		t.Fatal("期望因超过最大步数而报错，但没有报错")
	}
}
