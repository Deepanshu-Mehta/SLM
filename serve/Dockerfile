# Python 3.12 pinned (3.13 breaks parts of the HF stack). CPU-only torch keeps the image small.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Weights come from the model repo at build time: one canonical copy, no download on cold start.
ARG MODEL_REPO=Deepanshu027/slm-125m-base
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_REPO}', local_dir='/app/model', allow_patterns=['*.json', '*.safetensors'])"

COPY app.py .
RUN chown -R user:user /app
USER user
ENV MODEL_DIR=/app/model TORCH_THREADS=2 TOKENIZERS_PARALLELISM=false
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
