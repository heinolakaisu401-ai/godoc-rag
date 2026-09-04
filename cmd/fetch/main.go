// fetch 下载 Go 官方文档（markdown）作为 RAG 语料，保存到 data/docs。
// 运行：go run ./cmd/fetch
// 如果某个链接失效，手动把 markdown 文件放进 data/docs 目录即可，效果一样。
package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
)

// 语料清单：Go 官方仓库 doc/ 目录下的 markdown 文档（固定到某个发布版本保证稳定）。
var urls = []struct{ name, url string }{
	{"effective_go.md", "https://raw.githubusercontent.com/golang/go/go1.23.4/doc/effective_go.md"},
	{"comment.md", "https://raw.githubusercontent.com/golang/go/go1.23.4/doc/comment.md"},
	{"godebug.md", "https://raw.githubusercontent.com/golang/go/go1.23.4/doc/godebug.md"},
	{"README.md", "https://raw.githubusercontent.com/golang/go/go1.23.4/doc/README.md"},
}

func main() {
	outDir := flag.String("out", "data/docs", "输出目录")
	flag.Parse()

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatal(err)
	}

	for _, u := range urls {
		path := filepath.Join(*outDir, u.name)
		if err := download(u.url, path); err != nil {
			log.Printf("下载失败 %s: %v（可手动把 markdown 放进 %s）", u.name, err, *outDir)
			continue
		}
		log.Printf("已下载 %s", path)
	}
}

func download(url, path string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
