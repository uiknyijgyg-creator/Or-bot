# 🗒️ Manga Text Cleaner — OCR + LaMa

بوت Discord احترافي لحذف نصوص المانجا والمانهوا باستخدام:
- **manga-ocr** — نموذج OCR متخصص في المانجا اليابانية والكورية
- **LaMa Inpainting** — أفضل نموذج مفتوح المصدر لترميم الصور

---

## 🚀 التثبيت خطوة بخطوة

### 1. المتطلبات
- Python **3.10+**
- RAM: 4GB+ (8GB مستحسن)
- GPU: اختياري لكن يسرّع بشكل كبير

### 2. تثبيت المكتبات

```bash
# CPU فقط
pip install -r requirements.txt

# GPU (CUDA 11.8) — أسرع بكثير
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3. إعداد Discord Bot

1. اذهب إلى [Discord Developer Portal](https://discord.com/developers/applications)
2. **New Application** → أعطه اسماً
3. **Bot** → **Add Bot** → انسخ الـ **TOKEN**
4. فعّل **MESSAGE CONTENT INTENT**
5. **OAuth2** → **URL Generator**:
   - Scopes: `bot` + `applications.commands`
   - Permissions: `Send Messages` + `Attach Files` + `Read Message History`
6. افتح الرابط وأضف البوت لسيرفرك

### 4. تشغيل البوت

```bash
# Windows
set DISCORD_TOKEN=توكنك_هنا
python bot.py

# Linux / Mac
export DISCORD_TOKEN=توكنك_هنا
python bot.py
```

أو عدّل السطر في `bot.py`:
```python
DISCORD_TOKEN = "توكنك_هنا"
```

---

## 📋 الأوامر

| الأمر | الوصف |
|-------|--------|
| `/clean` | ارفع ZIP واختار النوع (مانجا/مانهوا) |
| `/cancel` | إلغاء عملية شغالة |
| `/stats` | إحصائياتك |
| `/help` | تعليمات البوت |

**أو:** ارفع ZIP مباشرة في الشات — يكتشفه تلقائياً!

---

## ⚙️ كيف يعمل

```
ZIP مضغوط
    ↓
فك الضغط + ترتيب الصور
    ↓
لكل صورة:
  OpenCV  →  يكتشف البالونات البيضاء
  manga-ocr → يتأكد إن فيها نص فعلاً
  LaMa    →  يمسح النص ويرمم الخلفية بذكاء
    ↓
ZIP جديد نظيف مرتب
```

---

## 📊 مقارنة مع الإصدار السابق (Gemini)

| | Gemini (السابق) | OCR + LaMa (الجديد) |
|---|---|---|
| دقة حذف النص | 70-85% | **90-95%** |
| ترميم الخلفية | OpenCV TELEA | **LaMa (أفضل بكثير)** |
| يحتاج API Key | ✅ نعم | ❌ لا |
| يعمل offline | ❌ | ✅ |
| سرعة (CPU) | سريع | أبطأ نسبياً |
| سرعة (GPU) | سريع | **سريع جداً** |

---

## 🔧 إعدادات متقدمة

في `bot.py`:

```python
CONCURRENCY    = 3    # صور تُعالج في نفس الوقت (زود لو عندك GPU)
THREAD_WORKERS = 4    # threads للمعالجة (= عدد CPU cores مستحسن)
SLICE_HEIGHT   = 1200 # ارتفاع شرائح المانهوا بالبكسل
```
