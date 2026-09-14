FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-opencv \
        git \
        ffmpeg \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python3 -m pip install --break-system-packages --upgrade pip \
    && grep -v opencv-python requirements.txt > req.txt \
    && python3 -m pip install --break-system-packages --no-cache-dir -r req.txt \
    && python3 -m pip install --break-system-packages --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install --break-system-packages --no-cache-dir git+https://github.com/cvg/LightGlue.git \
    && python3 -m pip install --break-system-packages --no-cache-dir "numpy<2" \
    && python3 -m pip uninstall --break-system-packages -y opencv-python

COPY . /app

CMD ["python3", "-m", "camera_fusion.launch", "configs/cam1.json", "configs/cam2.json"]