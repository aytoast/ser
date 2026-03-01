# Ethos Studio — HuggingFace Spaces Docker deployment
# Architecture: Python model (8000) + Node proxy (3000) + Next.js (3030)
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

# ─── Model layer: Python dependencies ────────────────────────────────────────
COPY model/voxtral-server/requirements.txt model/voxtral-server/requirements.txt
RUN pip install --no-cache-dir -r model/voxtral-server/requirements.txt

COPY model/voxtral-server/ model/voxtral-server/

# ─── Proxy server: Node dependencies ─────────────────────────────────────────
COPY demo/server/package*.json demo/server/
RUN cd demo/server && npm ci --omit=dev

COPY demo/server/ demo/server/

# ─── Frontend: Next.js build ──────────────────────────────────────────────────
COPY demo/package*.json demo/
RUN cd demo && npm ci

COPY demo/ demo/

# Build with empty API base → browser uses relative paths → nginx routes /api/*
RUN cd demo && NEXT_PUBLIC_API_URL="" npm run build \
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
