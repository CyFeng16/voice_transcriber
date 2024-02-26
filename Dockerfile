FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
WORKDIR /workspace
RUN apt-get update && \
    apt-get install --no-install-recommends -y software-properties-common git ffmpeg && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends -y python3.10 python3.10-venv python3.10-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    python3 -m pip install --upgrade pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-build-isolation flash_attn
COPY app.py func.py sample.mp3 /workspace/
RUN python -c "import app; app.model_init_wrapper('openai/whisper-large-v3'); app.model_init_wrapper('distil-whisper/large-v2')"
EXPOSE 49799
CMD ["python", "app.py"]
