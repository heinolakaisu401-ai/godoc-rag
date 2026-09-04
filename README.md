# Go 文档智能问答系统（RAG）

用 Go 实现的一个**检索增强生成（RAG）**问答系统，以 Go 官方文档为语料。
核心链路：**向量化 → 混合检索（向量 + 关键词）→ 重排 → 生成**，并内置一套
**LLM-as-judge 评测工具**，用数据驱动迭代。

## 技术栈

| 模块 | 技术 |
|---|---|
| 语言 / Web 框架 | Go + Gin |
| 向量数据库 | PostgreSQL + pgvector |
| 大模型（对话 + Embedding） | 通义千问 DashScope（OpenAI 兼容接口） |
| 评测 | LLM-as-judge |

## 目录结构

```
cmd/
  server/     HTTP 服务入口（POST /ask）
  ingest/     语料入库（切分 + 向量化 + 写库）
  eval/       评测工具（跑评测集 + 打分）
  fetch/      下载 Go 官方文档 markdown
internal/
  config/     配置加载
  llm/        OpenAI 兼容客户端（对话 + embedding）
  chunker/    文档切分
  store/      pgvector 存取
  rag/        RAG 主链路（检索/重排/生成）
  api/        Gin 路由与 handler
scripts/setup.sql   建表脚本
data/docs/          语料目录
data/eval/qa.jsonl  评测集
```

## 快速开始

### 0. 环境准备
- Go 1.22+
- PostgreSQL 14+，并安装 pgvector 扩展
- 通义千问 DashScope API Key（[申请地址](https://dashscope.console.aliyun.com)）

### 1. 配置
```bash
cp .env.example .env   # Windows 下手动复制改名，再填入 DASHSCOPE_API_KEY 和 DATABASE_URL
```

### 2. 建库建表
```bash
psql -U postgres -c "CREATE DATABASE godoc_rag;"
psql -U postgres -d godoc_rag -f scripts/setup.sql
```

### 3. 准备语料
```bash
go run ./cmd/fetch     # 下载 Go 官方文档到 data/docs（也可手动放 markdown）
```

### 4. 导入语料（切分 + 向量化 + 入库）
```bash
go run ./cmd/ingest -dir data/docs
```

### 5. 启动服务
```bash
go run ./cmd/server
```

### 6. 测试
```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"什么是 goroutine？"}'
```

### 7. 跑评测
```bash
go run ./cmd/eval -file data/eval/qa.jsonl
```

## 关键设计

- **混合检索**：向量检索（余弦相似度）+ 关键词检索（PostgreSQL 全文检索），用 RRF 融合。
- **重排**：检索先多召回（CandidateK=20），再用一次 LLM 调用对候选重排，只取 TopK=4 喂给生成。
- **可解释**：`/ask` 返回答案的同时返回引用来源（source + section + score），便于核查幻觉。
- **可评测**：`eval` 从「忠实度」和「检索相关性」两个维度打分，输出汇总分数，驱动迭代。
