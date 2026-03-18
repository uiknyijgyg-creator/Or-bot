# ══════════════════════════════════════════════════════════
#  Manga OCR Bot — Optimized Dockerfile (CPU-only)
#  حجم مُقلَّص: PyTorch CPU-only بدل الـ CUDA الكاملة
# ══════════════════════════════════════════════════════════

FROM python:3.11-slim

# ── مكتبات النظام الضرورية فقط ────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# ── 1) تثبيت PyTorch CPU-only أولاً ──────────
# CPU-only ~800MB بدل النسخة الكاملة ~2.5GB
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── 2) تثبيت باقي المكتبات ────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 3) تنظيف cache لتقليل الحجم ───────────────
RUN pip cache purge && \
    find /usr/local/lib/python3.11 -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ── 4) نسخ الكود ──────────────────────────────
COPY bot.py .

# ── متغيرات البيئة ─────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false

CMD ["python", "bot.py"]
