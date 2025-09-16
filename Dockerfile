# Use Ubuntu 22.04 LTS for better stability
FROM ubuntu:22.04

# Set environment variables (avoid warnings and optimize JAX)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies + Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        software-properties-common \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3-pip \
        python3.10-venv \
        libopenmpi-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 and pip point to Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip first
RUN python3 -m pip install --upgrade pip

# Create working directory
WORKDIR /app

# Install Python packages in order of dependencies
# First install core packages
RUN pip install --no-cache-dir setuptools wheel

# Install JAX (CPU version) - install before other packages that might need it
RUN pip install --no-cache-dir jax jaxlib

# Install Ray with specific components
RUN pip install --no-cache-dir "ray[rllib,default,serve]"

# Install gymnasium and gaming dependencies
RUN pip install --no-cache-dir "gymnasium[atari,accept-rom-license]" pygame

# Install Brax (should come after JAX)
RUN pip install --no-cache-dir brax

# Install JupyterLab
RUN pip install --no-cache-dir jupyterlab ipywidgets

# Create a non-root user for better security
RUN useradd -m -u 1000 jupyter && \
    chown -R jupyter:jupyter /app

# Switch to non-root user
USER jupyter

# Expose port for JupyterLab
EXPOSE 8888

# Create jupyter config directory
RUN mkdir -p /home/jupyter/.jupyter

# Launch JupyterLab when container starts
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.allow_origin='*'", "--ServerApp.base_url=/"]
