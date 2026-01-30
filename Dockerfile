# 使用 PyTorch 官方镜像（已包含 PyTorch 2.7.1, CUDA 12.6, cuDNN 9）
#
# 构建镜像（开发用，代码通过 docker-compose 挂载）：
# docker build -f Dockerfile -t x-anylabeling-server:dev .
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# 设置工作目录
WORKDIR /X-AnyLabeling-Server

# 设置环境变量（包含时区）
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 安装系统依赖（含 OpenCV 所需运行库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libxcb1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv 包管理器
RUN pip install --upgrade uv

# 安装项目依赖（安装所有可选依赖）
# PyTorch 已包含在基础镜像中，不需要单独安装
# 与 Rex-Omni 官方重叠的依赖固定版本，避免 transformers 等与 Rex-Omni 不兼容
RUN uv pip install --system \
        "fastapi[standard]>=0.115.0" \
        "pydantic>=2.12.0" \
        "openai>=1.99.1" \
        loguru \
        "numpy==1.26.4" \
        "requests>=2.26.0" \
        "opencv-python-headless>=4.11.0" \
        "Pillow==10.4.0" \
        black \
        flake8 \
        ultralytics \
        lapx \
        "transformers==4.51.3" \
        "accelerate==1.10.1" \
        "qwen_vl_utils==0.0.14" \
        decord \
        einops \
        "ftfy==6.1.1" \
        huggingface_hub \
        "hydra-core>=1.3.2" \
        "iopath>=0.1.10" \
        pandas \
        pycocotools \
        scikit-image \
        scikit-learn \
        "timm>=1.0.17" \
        typing_extensions \
        zai-sdk



