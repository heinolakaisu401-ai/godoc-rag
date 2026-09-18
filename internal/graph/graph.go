// Package graph 实现一个 LangGraph 风格的「状态图」编排引擎。
//
// 它把一次任务拆成多个节点（Node），用边（Edge）和条件边（Router）把节点
// 连成一张可分支、可循环的图，再由 Run 从入口节点驱动执行，直到路由到 End。
//
// 概念与 LangGraph 的对应关系：
//   - State    → LangGraph 的 state dict（节点间流转的共享状态）
//   - NodeFunc → 一个节点 = 一个改写 State 的函数
//   - Edge     → 固定边（from 执行完总是走到 to）
//   - Router   → 条件边（根据 State 决定下一个节点，支持循环）
//   - End      → 终止哨兵
//   - maxSteps → 循环保护，防止死循环
package graph

import (
	"context"
	"fmt"
)

// State 是节点间流转的共享状态。用 map[string]any 让引擎与业务解耦，
// 业务层按自己的 key 读写；等价于 LangGraph 的 state dict。
type State map[string]any

// End 是终止哨兵，路由到它表示本次运行结束。
const End = "__end__"

// NodeFunc 处理一个节点：读取并修改 State。返回 error 会中止整次运行。
type NodeFunc func(ctx context.Context, s State) error

// Router 是条件边的路由函数：根据 State 返回下一个节点的名字（或 End）。
type Router func(s State) string

// Graph 是一张尚未执行的状态图。
type Graph struct {
	nodes    map[string]NodeFunc
	edges    map[string]string // from -> to 的固定边
	routers  map[string]Router // from -> 条件路由
	entry    string
	maxSteps int
}

// New 创建一张空图，默认最多执行 25 步（循环保护）。
func New() *Graph {
	return &Graph{
		nodes:    map[string]NodeFunc{},
		edges:    map[string]string{},
		routers:  map[string]Router{},
		maxSteps: 25,
	}
}

// AddNode 注册一个节点。
func (g *Graph) AddNode(name string, fn NodeFunc) *Graph {
	g.nodes[name] = fn
	return g
}

// AddEdge 添加一条固定边：from 执行完总是走到 to。
func (g *Graph) AddEdge(from, to string) *Graph {
	g.edges[from] = to
	return g
}

// AddConditionalEdges 添加条件边：from 执行完后调用 router 决定下一个节点。
func (g *Graph) AddConditionalEdges(from string, router Router) *Graph {
	g.routers[from] = router
	return g
}

// SetEntryPoint 设置入口节点。
func (g *Graph) SetEntryPoint(name string) *Graph {
	g.entry = name
	return g
}

// SetMaxSteps 设置最大执行步数（循环保护），默认 25。
func (g *Graph) SetMaxSteps(n int) *Graph {
	if n > 0 {
		g.maxSteps = n
	}
	return g
}

// Run 从入口节点开始执行，直到路由到 End，返回最终的 State。
// 节点出错或步数超限都会中止并返回 error。
func (g *Graph) Run(ctx context.Context, initial State) (State, error) {
	if g.entry == "" {
		return nil, fmt.Errorf("图未设置入口节点")
	}
	s := State{}
	for k, v := range initial {
		s[k] = v
	}

	node := g.entry
	for step := 1; ; step++ {
		if node == End {
			return s, nil
		}
		if step > g.maxSteps {
			return nil, fmt.Errorf("图执行超过最大步数 %d，疑似死循环", g.maxSteps)
		}
		fn, ok := g.nodes[node]
		if !ok {
			return nil, fmt.Errorf("未知节点 %q", node)
		}
		if err := fn(ctx, s); err != nil {
			return nil, fmt.Errorf("节点 %q 执行失败: %w", node, err)
		}
		node = g.next(node, s)
	}
}

// next 决定当前节点执行完后的下一个节点：条件边优先于固定边，缺省终止。
func (g *Graph) next(node string, s State) string {
	if r, ok := g.routers[node]; ok {
		return r(s)
	}
	if to, ok := g.edges[node]; ok {
		return to
	}
	return End
}
