// Package config 负责从环境变量（以及可选的 .env 文件）加载配置。
package config

import (
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

// Config 汇总整个项目需要的配置项。
type Config struct {
	DashscopeAPIKey  string
	DashscopeBaseURL string
	LLMModel         string
	EmbedModel       string
	EmbedDim         int
	DatabaseURL      string
	ServerPort       string

	// RAG 链路的可调参数（方便用评测迭代：改 .env 即可，无需改代码）
	ChunkSize    int
	ChunkOverlap int
	TopK         int
	CandidateK   int
	UseRerank    bool
}

// Load 读取配置，.env 不存在时静默忽略（方便容器等场景直接注入环境变量）。
func Load() (*Config, error) {
	_ = godotenv.Load()

	return &Config{
		DashscopeAPIKey:  os.Getenv("DASHSCOPE_API_KEY"),
		DashscopeBaseURL: getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
		LLMModel:         getenv("LLM_MODEL", "qwen-plus"),
		EmbedModel:       getenv("EMBED_MODEL", "text-embedding-v3"),
		EmbedDim:         getenvInt("EMBED_DIM", 1024),
		DatabaseURL:      getenv("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/godoc_rag?sslmode=disable"),
		ServerPort:       getenv("SERVER_PORT", "8080"),

		ChunkSize:    getenvInt("CHUNK_SIZE", 800),
		ChunkOverlap: getenvInt("CHUNK_OVERLAP", 100),
		TopK:         getenvInt("RAG_TOP_K", 4),
		CandidateK:   getenvInt("RAG_CANDIDATE_K", 20),
		UseRerank:    getenvBool("RAG_USE_RERANK", true),
	}, nil
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func getenvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func getenvBool(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}
