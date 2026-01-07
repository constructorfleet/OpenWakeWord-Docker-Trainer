FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV OUTPUT_DIR="/workspace/out"
ENV ROOT_PATH="/workspace"
ENV DATA_PATH="${ROOT_PATH}/data"
ENV GENERATED_PATH="${DATA_PATH}/generated"
ENV RIRS_PATH="${DATA_PATH}/rirs"
ENV AUDIOSET_PATH="${DATA_PATH}/audioset"
ENV AUDIOSET16K_PATH="${DATA_PATH}/audioset_16k"
ENV FSD50K_PATH="${DATA_PATH}/fsd50k"
ENV FSD50K16K_PATH="${DATA_PATH}/fsd50k_16k"
ENV MUSAN_PATH="${DATA_PATH}/musan"
ENV MUSAN16K_PATH="${DATA_PATH}/musan_16k"
ENV FMA_PATH="${DATA_PATH}/fma"
ENV FMA16K_PATH="${DATA_PATH}/fma_16k"
ENV FEATURES_PATH="${DATA_PATH}/features"
ENV CONFIG_FILE="${ROOT_PATH}/configs/model.yaml"
ENV RUNTIME_CONFIG_FILE="${ROOT_PATH}/configs/model.runtime.yaml"
ENV POSITIVES_PATH="${DATA_PATH}/positives"

# ---- system deps ----
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    sox \
    libsndfile1 \
    libgl1 \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

# ---- pip ----
RUN python -m pip install --upgrade pip setuptools wheel

# ---- PyTorch CUDA ----
RUN pip install \
    torch==2.9.1 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# ---- core ML/audio stack ----
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ---- torchcodec CUDA ----
RUN pip install torchcodec --index-url https://download.pytorch.org/whl/cu126

# ---- openWakeWord ----
WORKDIR /workspace
RUN git clone https://github.com/dscripka/openWakeWord.git
RUN pip install -e openWakeWord

# ---- runtime paths ----
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
COPY . .
# ---- default ----
CMD ["bash"]
