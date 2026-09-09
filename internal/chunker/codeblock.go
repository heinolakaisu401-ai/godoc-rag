package chunker

import "strings"

// CodeBlockStatus 描述一段文本中代码块的完整性。
type CodeBlockStatus int

const (
	NoCode        CodeBlockStatus = iota // 没有围栏标记
	Complete                             // 所有围栏成对，段尾不处于代码块内
	DanglingOpen                         // 有未闭合的开头围栏，缺结尾
	DanglingClose                        // 只有孤立的结尾围栏，缺开头
)

// BlockInfo 是单个代码块的完整性判定结果。
type BlockInfo struct {
	Status CodeBlockStatus
	Lang   string // 语言标记（如 go / bash）；裸围栏或闭合围栏为空
	Start  int    // 起始行号（1 基）
	End    int    // 结束行号（1 基）
}

// CheckCodeBlocks 扫描一段文本，返回其中每个代码块的完整性判定。
//
// 判定约定（贴合 Go 官方文档的围栏写法）：
//   - 带语言信息的围栏（如 ```go）视为开围栏；
//   - 不带语言信息的裸围栏（```）视为闭围栏；
//   - 段尾仍处于代码块内 => 该块 DanglingOpen（缺结尾）；
//   - 未处于代码块内却出现裸围栏 => DanglingClose（缺开头）。
func CheckCodeBlocks(text string) []BlockInfo {
	var blocks []BlockInfo
	var cur *BlockInfo

	lines := strings.Split(text, "\n")
	for i, line := range lines {
		if !isFenceLine(line) {
			continue
		}
		lang := fenceLang(line)
		if cur == nil {
			// 不在代码块内
			if lang == "" {
				// 裸围栏且无语言信息：在 Go 文档里通常是某段代码的闭合围栏
				blocks = append(blocks, BlockInfo{Status: DanglingClose, Lang: "", Start: i + 1, End: i + 1})
				continue
			}
			cur = &BlockInfo{Status: Complete, Lang: lang, Start: i + 1}
		} else {
			// 在代码块内：这是闭合围栏
			cur.End = i + 1
			cur.Status = Complete
			blocks = append(blocks, *cur)
			cur = nil
		}
	}
	if cur != nil {
		cur.Status = DanglingOpen
		blocks = append(blocks, *cur)
	}
	return blocks
}

// Summarize 把逐块结果聚合为单个状态，用于“这段要不要贴警告”的粗粒度判断。
// 规则：最危险者胜出（DanglingOpen > DanglingClose > Complete > NoCode）。
func Summarize(blocks []BlockInfo) CodeBlockStatus {
	worst := NoCode
	for _, b := range blocks {
		if severity(b.Status) > severity(worst) {
			worst = b.Status
		}
	}
	return worst
}

func severity(s CodeBlockStatus) int {
	switch s {
	case DanglingOpen:
		return 3
	case DanglingClose:
		return 2
	case Complete:
		return 1
	default:
		return 0
	}
}

// fenceLang 提取围栏行里的语言标记，例如 "```go" -> "go"，"```" -> ""。
func fenceLang(line string) string {
	t := strings.TrimSpace(line)
	t = strings.TrimLeft(t, "`~")
	return strings.TrimSpace(t)
}
