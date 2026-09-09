package chunker

import (
	"strings"
	"testing"
)

// 代码块内存在空行和 # 时，也必须作为一个原子单元，不被切碎。
func TestSplitKeepsCodeBlockAtomic(t *testing.T) {
	c := New(800, 100)
	text := "## select\n\n说明文字。\n\n```go\nch1 := make(chan int)\nch2 := make(chan string)\n\nselect {\ncase v := <-ch1:\n    fmt.Println(v)\ndefault:\n    fmt.Println(\"无数据\")\n}\n```\n\n结尾段落。\n"
	chunks := c.Split(text)

	codeChunks := 0
	for _, ch := range chunks {
		if !strings.Contains(ch.Text, "```go") {
			continue
		}
		codeChunks++
		if !strings.Contains(ch.Text, "fmt.Println(v)") {
			t.Errorf("代码块内容不完整: %q", ch.Text)
		}
		if !strings.HasSuffix(strings.TrimSpace(ch.Text), "```") {
			t.Errorf("代码块缺少闭合围栏: %q", ch.Text)
		}
	}
	if codeChunks != 1 {
		t.Fatalf("期望代码块被切成 1 个 chunk，实际 %d 个", codeChunks)
	}
}

// 硬切应优先落在换行符处，不切断一行。
func TestSplitLongPrefersNewlineBoundary(t *testing.T) {
	c := New(10, 0)
	s := "abcde\nfghij\nklmno\npqrst\n"
	parts := c.splitLong(s)
	if len(parts) < 2 {
		t.Fatalf("期望被切分成多块，实际 %d 块", len(parts))
	}
	for i, p := range parts {
		if i < len(parts)-1 && !strings.HasSuffix(p, "\n") {
			t.Errorf("第 %d 块没有在换行处断开: %q", i, p)
		}
	}
}

func TestCheckCodeBlocks_Complete(t *testing.T) {
	blocks := CheckCodeBlocks("```go\nfmt.Println(\"hi\")\n```\n")
	if len(blocks) != 1 || blocks[0].Status != Complete || blocks[0].Lang != "go" {
		t.Fatalf("期望 [Complete(go)]，实际 %+v", blocks)
	}
}

func TestCheckCodeBlocks_DanglingOpen(t *testing.T) {
	blocks := CheckCodeBlocks("```go\nfmt.Println(\"hi\")\n")
	if len(blocks) != 1 || blocks[0].Status != DanglingOpen {
		t.Fatalf("期望 [DanglingOpen]，实际 %+v", blocks)
	}
	if blocks[0].Lang != "go" {
		t.Fatalf("期望 lang=go，实际 %q", blocks[0].Lang)
	}
}

func TestCheckCodeBlocks_DanglingClose(t *testing.T) {
	blocks := CheckCodeBlocks("    return result\n```\n")
	if len(blocks) != 1 || blocks[0].Status != DanglingClose {
		t.Fatalf("期望 [DanglingClose]，实际 %+v", blocks)
	}
}

func TestCheckCodeBlocks_Mixed(t *testing.T) {
	text := "```go\nfmt.Println(\"hi\")\n```\n\n```python\nimport os\n"
	blocks := CheckCodeBlocks(text)
	if len(blocks) != 2 {
		t.Fatalf("期望 2 块，实际 %d: %+v", len(blocks), blocks)
	}
	if blocks[0].Status != Complete || blocks[0].Lang != "go" {
		t.Errorf("第 1 块应为 Complete(go)，实际 %+v", blocks[0])
	}
	if blocks[1].Status != DanglingOpen || blocks[1].Lang != "python" {
		t.Errorf("第 2 块应为 DanglingOpen(python)，实际 %+v", blocks[1])
	}
	if got := Summarize(blocks); got != DanglingOpen {
		t.Errorf("Summarize 期望 DanglingOpen，实际 %v", got)
	}
}

func TestCheckCodeBlocks_NoCode(t *testing.T) {
	if got := Summarize(CheckCodeBlocks("普通文本，没有代码块。")); got != NoCode {
		t.Fatalf("期望 NoCode，实际 %v", got)
	}
}
