# Dockerfile

# ===== 第一阶段：构建 =====
# 从 Python 官方镜像开始
FROM python:3.12-slim AS builder

# 设置工作目录（容器里的路径）
WORKDIR /app

# 先把 requirements.txt 拷贝进去
# 这一步单独做是为了利用 Docker 的缓存机制
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt


# ===== 第二阶段：运行 =====
FROM python:3.12-slim

WORKDIR /app

# 从构建阶段复制安装好的包
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# 复制项目代码
COPY . .

# 声明容器运行时会监听 8000 端口
EXPOSE 8000

# 容器启动时运行的命令
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
