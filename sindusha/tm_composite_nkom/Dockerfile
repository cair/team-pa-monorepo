FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04

# Prevent tzdata questions
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set the working directory
WORKDIR /workspace

# Set Python path to include src directory
ENV PYTHONPATH="/workspace/src:${PYTHONPATH}"