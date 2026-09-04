# ---------- 构建阶段 ----------
FROM golang:1.26-alpine AS builder

WORKDIR /app

# 国内加速拉取依赖 + 静态编译
ENV GOPROXY=https://goproxy.cn,direct \
    CGO_ENABLED=0 \
    GOOS=linux

# 先复制依赖清单，利用 Docker 层缓存
COPY go.mod go.sum ./
RUN go mod download

# 再复制源码并编译
COPY . .
RUN go build -o /godoc-server ./cmd/server

# ---------- 运行阶段 ----------
FROM alpine:3.20

# ca-certificates：调用 dashscope 的 HTTPS 接口需要
RUN apk add --no-cache ca-certificates tzdata

WORKDIR /app
COPY --from=builder /godoc-server /godoc-server

EXPOSE 8080
ENTRYPOINT ["/godoc-server"]
