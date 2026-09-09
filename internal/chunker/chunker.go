// Package chunker 负责把长文档切成适合做检索和生成的小块（chunk）。
package chunker

import (
	"regexp"
	"strings"
)

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

// headingRe 匹配 ATX 标题（1~6 个 # 后跟至少一个空白）。
var headingRe = regexp.MustCompile(`^#{1,6}\s+`)

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
//
// 切分边界优先级（从高到低）：
//  1. fenced code block（``` 围栏）——整个代码块作为原子单元，块内不切分；
//  2. markdown 标题（#）；
//  3. 空行（段落结束）；
//  4. 超过 MaxChars 的段落按字符窗口硬切（带 Overlap 重叠，且优先落在换行符处）。
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

	inCode := false
	for _, line := range strings.Split(text, "\n") {
		if isFenceLine(line) {
			if inCode {
				// 关闭围栏：代码块结束，把围栏行写进 buf 后整块 flush。
				buf.WriteString(line)
				buf.WriteString("\n")
				inCode = false
				flush()
			} else {
				// 开启围栏：先封存前面的段落，再开始新的代码块。
				flush()
				buf.WriteString(line)
				buf.WriteString("\n")
				inCode = true
			}
			continue
		}

		if inCode {
			// 代码块内：空行、# 等一律原样保留，不做任何切分。
			buf.WriteString(line)
			buf.WriteString("\n")
			continue
		}

		trimmed := strings.TrimSpace(line)
		if headingRe.MatchString(trimmed) { // markdown 标题：开启新段落，并记录标题
			flush()
			section = strings.TrimSpace(headingRe.ReplaceAllString(trimmed, ""))
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
// 避免关键句子被切断后信息丢失。断点优先落在换行符处，避免切断一行代码。
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
		// 若非最后一块，优先把断点移到最近的换行符之后，避免切断一行。
		if end < len(runes) {
			if idx := lastNewline(runes, start, end); idx > start {
				end = idx + 1
			}
		}
		out = append(out, string(runes[start:end]))
		if end >= len(runes) {
			break
		}
	}
	return out
}

// lastNewline 返回 runes[start:end] 范围内最后一个 '\n' 的下标，找不到返回 -1。
func lastNewline(runes []rune, start, end int) int {
	for i := end - 1; i >= start; i-- {
		if runes[i] == '\n' {
			return i
		}
	}
	return -1
}

// isFenceLine 判断一行是否为 fenced code block 的围栏标记：
// 最多 3 个前导空格之后，紧跟 3 个或以上反引号或波浪线。
func isFenceLine(line string) bool {
	t := line
	spaces := 0
	for spaces < len(t) && t[spaces] == ' ' {
		spaces++
	}
	if spaces > 3 {
		return false // 缩进超过 3 个空格是缩进代码块，不是围栏
	}
	t = t[spaces:]
	return strings.HasPrefix(t, "```") || strings.HasPrefix(t, "~~~")
}
