# Ethos Studio — HuggingFace Spaces Docker deployment
# Architecture: Python API (8000) + Node proxy (3000) + Next.js (3030)
#               nginx on port 7860 routes traffic between layers

FROM python:3.11-slim

# ─── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl nginx supervisor \
    && rm -rf /var/lib/apt/lists/*

# Node.js 22
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── API layer: Python dependencies ──────────────────────────────────────────
# CACHE_BUST: force pip layer rebuild when bumping onnxruntime version
ARG CACHE_BUST=20260302
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ api/

# ─── Proxy server: Node dependencies ─────────────────────────────────────────
COPY proxy/package*.json proxy/
RUN cd proxy && npm ci --omit=dev

COPY proxy/ proxy/

# ─── Frontend: Next.js build ──────────────────────────────────────────────────
COPY web/package*.json web/
RUN cd web && npm ci

COPY web/ web/

# Build with empty API base → browser uses relative paths → nginx routes /api/*
RUN cd web && NEXT_PUBLIC_API_URL="" npm run build \
    && cp -r public .next/standalone/public \
    && cp -r .next/static .next/standalone/.next/static

# ─── FER model (MobileViT-XXS ONNX, 8-class facial emotion) ─────────────────
COPY models/ models/

# ─── Process manager + reverse proxy config ───────────────────────────────────
COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisor/conf.d/app.conf

# HuggingFace Spaces public port
EXPOSE 7860

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/app.conf"]
