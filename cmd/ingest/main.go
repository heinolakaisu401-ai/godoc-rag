// ingest 把 data/docs 下的 markdown 文档切成 chunk、向量化并写入数据库。
// 运行：go run ./cmd/ingest -dir data/docs
//       或 go run ./cmd/ingest -file data/docs/effective_go.md
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"path/filepath"
	"strings"

	"godoc-rag/internal/chunker"
	"godoc-rag/internal/config"
	"godoc-rag/internal/llm"
	"godoc-rag/internal/store"
)

func main() {
	dir := flag.String("dir", "data/docs", "要导入的文档目录")
	file := flag.String("file", "", "导入单个文件（优先级高于 -dir）")
	flag.Parse()

	cfg, err := config.Load()
	if err != nil {
		log.Fatal(err)
	}
	if cfg.DashscopeAPIKey == "" {
		log.Fatal("缺少 DASHSCOPE_API_KEY，请先配置 .env")
	}

	ctx := context.Background()

	st, err := store.Open(ctx, cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("连接数据库失败: %v", err)
	}
	defer st.Close()

	llmClient := llm.New(cfg.DashscopeAPIKey, cfg.DashscopeBaseURL)
	ck := chunker.New(cfg.ChunkSize, cfg.ChunkOverlap) // 块大小和重叠从 .env 读取

	var files []string
	if *file != "" {
		files = []string{*file}
	} else {
		files, err = listMarkdown(*dir)
		if err != nil {
			log.Fatal(err)
		}
	}
	if len(files) == 0 {
		log.Fatalf("没有找到可导入的 markdown 文件（请先运行 go run ./cmd/fetch 下载语料）")
	}

	total := 0
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			log.Printf("跳过 %s: %v", f, err)
			continue
		}
		chunks := ck.Split(string(data))
		for _, ch := range chunks {
			emb, err := llmClient.Embed(ctx, cfg.EmbedModel, ch.Text, cfg.EmbedDim)
			if err != nil {
				log.Fatalf("向量化 %s 失败: %v", f, err)
			}
			if err := st.Insert(ctx, filepath.Base(f), ch.Section, ch.Text, emb); err != nil {
				log.Fatalf("写入 %s 失败: %v", f, err)
			}
			total++
		}
		log.Printf("已导入 %s：%d 个 chunk", f, len(chunks))
	}
	log.Printf("完成，共导入 %d 个 chunk", total)
}

func listMarkdown(dir string) ([]string, error) {
	var files []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(strings.ToLower(path), ".md") {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}
