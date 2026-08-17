# Dockerfile

# ===== 第一阶段：构建 =====
# 基础镜像走国内源（daocloud 镜像），避免 Docker Hub 拉取超时
FROM docker.m.daocloud.io/library/python:3.12-slim AS builder

WORKDIR /app

# 先拷贝 requirements.txt，利用 Docker 缓存机制
COPY requirements.txt .

# pip 走阿里云源，加速依赖安装
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt


# ===== 第二阶段：运行 =====
FROM docker.m.daocloud.io/library/python:3.12-slim

WORKDIR /app

# 从构建阶段复制安装好的依赖包
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# 复制项目代码
COPY . .

# 声明容器运行时会监听 8000 端口
EXPOSE 8000

# 用 python -m uvicorn 启动（镜像只复制了 site-packages，未复制可执行文件）
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
