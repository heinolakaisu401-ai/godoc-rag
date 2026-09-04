// Package chunker 负责把长文档切成适合做检索和生成的小块（chunk）。
package chunker

import "strings"

// Chunk 是入库的最小文本单元。
type Chunk struct {
	Text    string // 文本内容
	Section string // 所属标题（markdown 的 # 标题）
	Index   int    // 在整份语料中的序号
}

// Chunker 按标题 + 段落切分，并对过长内容做带重叠的硬切。
type Chunker struct {
	MaxChars int
	Overlap  int
}

// New 创建切分器。maxChars 为每个 chunk 的目标长度，overlap 为相邻 chunk 的重叠字符数。
func New(maxChars, overlap int) *Chunker {
	if maxChars <= 0 {
		maxChars = 800
	}
	if overlap < 0 {
		overlap = 0
	}
	return &Chunker{MaxChars: maxChars, Overlap: overlap}
}

// Split 把 markdown 文本切分成多个 Chunk。
func (c *Chunker) Split(text string) []Chunk {
	var chunks []Chunk
	var section string

	var buf strings.Builder
	flush := func() {
		s := strings.TrimSpace(buf.String())
		buf.Reset()
		if s == "" {
			return
		}
		for _, part := range c.splitLong(s) {
			chunks = append(chunks, Chunk{Text: part, Section: section, Index: len(chunks)})
		}
	}

	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") { // markdown 标题：开启新段落，并记录标题
			flush()
			section = strings.TrimLeft(trimmed, "# ")
			continue
		}
		if trimmed == "" { // 空行：段落结束
			flush()
			continue
		}
		buf.WriteString(line)
		buf.WriteString("\n")
	}
	flush()
	return chunks
}

// splitLong 把超长文本按 MaxChars 硬切，相邻块之间保留 Overlap 个字符的重叠，
// 避免关键句子被切断后信息丢失。
func (c *Chunker) splitLong(s string) []string {
	runes := []rune(s)
	if len(runes) <= c.MaxChars {
		return []string{s}
	}
	step := c.MaxChars - c.Overlap
	if step <= 0 {
		step = c.MaxChars
	}

	var out []string
	for start := 0; start < len(runes); start += step {
		end := start + c.MaxChars
		if end > len(runes) {
			end = len(runes)
		}
		out = append(out, string(runes[start:end]))
		if end == len(runes) {
			break
		}
	}
	return out
}
