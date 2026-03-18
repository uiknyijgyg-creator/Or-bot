FROM python:3.11-slim

# ── مكتبات النظام المطلوبة لـ OpenCV ──────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── تثبيت المكتبات أولاً (cache layer) ────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── نسخ الكود ─────────────────────────────────
COPY bot.py .

# ── تشغيل البوت ───────────────────────────────
CMD ["python", "bot.py"]
