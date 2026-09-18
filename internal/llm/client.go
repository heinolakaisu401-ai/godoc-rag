// Package llm 封装了一个通用的 OpenAI 兼容接口客户端，
// 同时用于通义千问（DashScope）的对话补全和文本向量化两个能力。
// 之所以写成 OpenAI 兼容格式，是为了以后想换服务商时只改配置、不改代码。
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client 是 OpenAI 兼容接口的客户端。
type Client struct {
	apiKey  string
	baseURL string
	http    *http.Client
}

// New 创建一个客户端。baseURL 形如 https://dashscope.aliyuncs.com/compatible-mode/v1
func New(apiKey, baseURL string) *Client {
	return &Client{
		apiKey:  apiKey,
		baseURL: baseURL,
		http:    &http.Client{Timeout: 90 * time.Second},
	}
}

// Message 是一条对话消息。兼容工具调用：
//   - assistant 消息可带 ToolCalls（模型要求调用工具）；
//   - tool 消息用 ToolCallID 关联到对应调用、Name 为工具名、Content 为结果。
type Message struct {
	Role       string     `json:"role"` // system / user / assistant / tool
	Content    string     `json:"content,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

// ToolCall 是模型要求调用的一次工具调用。
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"` // "function"
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction 是被调用的函数名与参数（JSON 字符串）。
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool 是给模型看的工具定义（OpenAI function calling 协议）。
type Tool struct {
	Type     string      `json:"type"` // "function"
	Function FunctionDef `json:"function"`
}

// FunctionDef 描述一个函数工具的名称、说明与 JSON Schema 参数。
type FunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters,omitempty"`
}

// ---------- 对话补全 ----------

type chatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature,omitempty"`
	Tools       []Tool    `json:"tools,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
}

type apiError struct {
	Message string `json:"message"`
	Code    string `json:"code"`
}

type chatResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
	Error *apiError `json:"error"`
}

// Chat 发送一轮对话，返回助手的回复文本（不带工具）。
func (c *Client) Chat(ctx context.Context, model string, messages []Message) (string, error) {
	m, err := c.chatMessage(ctx, model, messages, nil)
	if err != nil {
		return "", err
	}
	return m.Content, nil
}

// ChatWithTools 发送一轮对话并附带工具定义，返回完整 assistant 消息
// （可能携带 ToolCalls，表示模型要求调用工具）。
func (c *Client) ChatWithTools(ctx context.Context, model string, messages []Message, tools []Tool) (Message, error) {
	return c.chatMessage(ctx, model, messages, tools)
}

// chatMessage 是对话补全的公共实现。
func (c *Client) chatMessage(ctx context.Context, model string, messages []Message, tools []Tool) (Message, error) {
	body := chatRequest{Model: model, Messages: messages, Temperature: 0.2, Tools: tools}

	raw, err := c.do(ctx, "/chat/completions", body)
	if err != nil {
		return Message{}, err
	}
	var out chatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return Message{}, err
	}
	if out.Error != nil {
		return Message{}, fmt.Errorf("llm api error: %s", out.Error.Message)
	}
	if len(out.Choices) == 0 {
		return Message{}, errors.New("llm api 返回了空的 choices")
	}
	return out.Choices[0].Message, nil
}

// ---------- 文本向量化 ----------

type embedRequest struct {
	Model      string `json:"model"`
	Input      string `json:"input"`
	Dimensions int    `json:"dimensions,omitempty"`
}

type embedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Error *apiError `json:"error"`
}

// Embed 把一段文本转换成向量。dim 为向量维度（需与数据库列一致）。
func (c *Client) Embed(ctx context.Context, model, text string, dim int) ([]float32, error) {
	body := embedRequest{Model: model, Input: text, Dimensions: dim}

	raw, err := c.do(ctx, "/embeddings", body)
	if err != nil {
		return nil, err
	}
	var out embedResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	if out.Error != nil {
		return nil, fmt.Errorf("embedding api error: %s", out.Error.Message)
	}
	if len(out.Data) == 0 {
		return nil, errors.New("embedding api 返回了空的 data")
	}
	return out.Data[0].Embedding, nil
}

// newRequest 构造一个带鉴权的 POST 请求。
func (c *Client) newRequest(ctx context.Context, path string, payload any) (*http.Request, error) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	return req, nil
}

// do 发起一次 POST 请求并返回原始响应体。
func (c *Client) do(ctx context.Context, path string, payload any) ([]byte, error) {
	req, err := c.newRequest(ctx, path, payload)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("api 返回状态 %d: %s", resp.StatusCode, string(data))
	}
	return data, nil
}

// streamChunk 是流式（SSE）响应里的一小段增量。
type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
}

// StreamChat 以流式方式调用模型：每收到一段文本增量就回调 onDelta，
// 同时累积可能出现的工具调用，返回聚合后的完整 assistant 消息。
// 说明：模型调用工具的那几轮通常没有文本增量，所以 onDelta 只在最终作答时才真正吐字。
func (c *Client) StreamChat(ctx context.Context, model string, messages []Message, tools []Tool, onDelta func(string) error) (Message, error) {
	req, err := c.newRequest(ctx, "/chat/completions", chatRequest{
		Model: model, Messages: messages, Temperature: 0.2, Tools: tools, Stream: true,
	})
	if err != nil {
		return Message{}, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return Message{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return Message{}, fmt.Errorf("api 返回状态 %d: %s", resp.StatusCode, string(data))
	}

	var full string
	toolCalls := map[int]*ToolCall{}
	maxIdx := -1
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var ch streamChunk
		if err := json.Unmarshal([]byte(data), &ch); err != nil {
			continue // 忽略无法解析的行
		}
		for _, choice := range ch.Choices {
			d := choice.Delta
			if d.Content != "" {
				full += d.Content
				if onDelta != nil {
					if err := onDelta(d.Content); err != nil {
						return Message{}, err
					}
				}
			}
			for _, tc := range d.ToolCalls {
				cur, ok := toolCalls[tc.Index]
				if !ok {
					cur = &ToolCall{Type: "function"}
					toolCalls[tc.Index] = cur
				}
				if tc.ID != "" {
					cur.ID = tc.ID
				}
				if tc.Function.Name != "" {
					cur.Function.Name = tc.Function.Name
				}
				cur.Function.Arguments += tc.Function.Arguments
				if tc.Index > maxIdx {
					maxIdx = tc.Index
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return Message{}, err
	}

	msg := Message{Role: "assistant", Content: full}
	for i := 0; i <= maxIdx; i++ {
		if tc, ok := toolCalls[i]; ok {
			msg.ToolCalls = append(msg.ToolCalls, *tc)
		}
	}
	return msg, nil
}
