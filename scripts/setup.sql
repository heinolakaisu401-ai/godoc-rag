-- =============================================================
-- RAG 项目的数据库初始化脚本
-- 执行方式（在项目根目录）：
--   psql -U postgres -c "CREATE DATABASE godoc_rag;"     （首次建库，只需一次）
--   psql -U postgres -d godoc_rag -f scripts/setup.sql
-- =============================================================

-- 1) 启用 pgvector 扩展（需要先安装 pgvector，见 README）
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) 建 chunks 表
-- 注意：vector(1024) 的维度必须和 .env 里 EMBED_DIM 一致（默认 1024）
CREATE TABLE IF NOT EXISTS chunks (
    id         BIGSERIAL PRIMARY KEY,
    source     TEXT NOT NULL,                 -- 来源文件名
    section    TEXT NOT NULL DEFAULT '',      -- 所属 markdown 标题
    content    TEXT NOT NULL,                 -- 文本内容
    embedding  vector(1024) NOT NULL,         -- 向量
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3) 向量检索索引（HNSW，余弦距离）
-- 如果你的 pgvector 版本较老（< 0.5.0）不支持 HNSW，改用下面注释里的 IVFFlat：
--   CREATE INDEX chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- 4) 关键词检索索引（英文全文检索，配合混合检索使用）
CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
    ON chunks USING gin (to_tsvector('english', content));
