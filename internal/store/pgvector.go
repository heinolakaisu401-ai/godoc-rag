// Package store 封装对 PostgreSQL + pgvector 的访问。
// 采用 database/sql 接口 + pgx 的 stdlib 适配器 + pgvector-go 的向量类型。
package store

import (
	"context"
	"database/sql"
	"fmt"

	_ "github.com/jackc/pgx/v5/stdlib" // 注册 pgx 驱动
	"github.com/pgvector/pgvector-go"
)

// Chunk 是存储在数据库里的一段文本 + 其向量。
type Chunk struct {
	ID        int64
	Source    string
	Section   string
	Content   string
	Embedding []float32
}

// Retrieved 是检索返回的一条结果，Score 越高越相关。
type Retrieved struct {
	ID      int64
	Source  string
	Section string
	Content string
	Score   float64
}

// Store 持有一个数据库连接池。
type Store struct {
	db *sql.DB
}

// Open 建立连接并 ping 验证。
func Open(ctx context.Context, dsn string) (*Store, error) {
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		return nil, err
	}
	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("连接数据库失败（请确认 PostgreSQL 已启动、DATABASE_URL 正确）: %w", err)
	}
	return &Store{db: db}, nil
}

// Close 释放连接池。
func (s *Store) Close() error { return s.db.Close() }

// Insert 写入一条 chunk 及其向量。
func (s *Store) Insert(ctx context.Context, source, section, content string, emb []float32) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO chunks (source, section, content, embedding) VALUES ($1, $2, $3, $4)`,
		source, section, content, pgvector.NewVector(emb),
	)
	return err
}

// VectorSearch 用余弦相似度做向量检索，返回按相似度降序的结果。
// 注意：pgvector 的 <=> 是余弦「距离」，越小越相似，所以用 1-距离 得到相似度。
func (s *Store) VectorSearch(ctx context.Context, emb []float32, limit int) ([]Retrieved, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, source, section, content, 1 - (embedding <=> $1) AS score
		 FROM chunks
		 ORDER BY embedding <=> $1
		 LIMIT $2`,
		pgvector.NewVector(emb), limit,
	)
	if err != nil {
		return nil, err
	}
	return scanRetrieved(rows)
}

// KeywordSearch 用 PostgreSQL 全文检索做关键词搜索（BM25 思想的近似）。
// 与向量检索互补，构成混合检索的两路。
func (s *Store) KeywordSearch(ctx context.Context, query string, limit int) ([]Retrieved, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, source, section, content, ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) AS score
		 FROM chunks
		 WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
		 ORDER BY score DESC
		 LIMIT $2`,
		query, limit,
	)
	if err != nil {
		return nil, err
	}
	return scanRetrieved(rows)
}

func scanRetrieved(rows *sql.Rows) ([]Retrieved, error) {
	defer rows.Close()
	var out []Retrieved
	for rows.Next() {
		var r Retrieved
		if err := rows.Scan(&r.ID, &r.Source, &r.Section, &r.Content, &r.Score); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}
