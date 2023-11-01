FROM nvcr.io/nvidia/pytorch:23.08-py3

LABEL maintainer="Ben Sanders"

WORKDIR /app

ADD . /app

RUN apt-get update && apt-get install -y \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Use a virtual environment
# This ensures that our application is isolated from the system's python
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python3 -m venv ../.venv && . ../.venv/bin/activate
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "generate_images.py"]