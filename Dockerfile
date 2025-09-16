# Sử dụng Python 3.10 (ổn định với Ray và JAX)
FROM python:3.10-slim

# Thiết lập biến môi trường (tránh cảnh báo và optimize JAX)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        python3-dev \
        libopenmpi-dev \
        && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Cài đặt Ray RLlib và các phụ thuộc
RUN pip install --upgrade pip setuptools wheel && \
    pip install "ray[rllib]" "ray[default]" "ray[serve]" && \
    pip install gymnasium[atari,accept-rom-license] && \
    pip install pygame

# Cài đặt JAX + Brax (CPU version)
RUN pip install --upgrade jax jaxlib brax

# Cài đặt JupyterLab
RUN pip install jupyterlab

# Expose port cho JupyterLab
EXPOSE 8888

# Khởi động JupyterLab khi container chạy
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]

