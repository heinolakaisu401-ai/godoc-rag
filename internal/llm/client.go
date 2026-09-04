// Package llm 封装了一个通用的 OpenAI 兼容接口客户端，
// 同时用于通义千问（DashScope）的对话补全和文本向量化两个能力。
// 之所以写成 OpenAI 兼容格式，是为了以后想换服务商时只改配置、不改代码。
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
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

// Message 是一条对话消息。
type Message struct {
	Role    string `json:"role"` // system / user / assistant
	Content string `json:"content"`
}

// ---------- 对话补全 ----------

type chatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature,omitempty"`
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

// Chat 发送一轮对话，返回助手的回复文本。
func (c *Client) Chat(ctx context.Context, model string, messages []Message) (string, error) {
	body := chatRequest{Model: model, Messages: messages, Temperature: 0.2}

	raw, err := c.do(ctx, "/chat/completions", body)
	if err != nil {
		return "", err
	}
	var out chatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", err
	}
	if out.Error != nil {
		return "", fmt.Errorf("llm api error: %s", out.Error.Message)
	}
	if len(out.Choices) == 0 {
		return "", errors.New("llm api 返回了空的 choices")
	}
	return out.Choices[0].Message.Content, nil
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

// do 发起一次 POST 请求并返回原始响应体。
func (c *Client) do(ctx context.Context, path string, payload any) ([]byte, error) {
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
