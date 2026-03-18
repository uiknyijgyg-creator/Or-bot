"""
╔══════════════════════════════════════════════════════════╗
║           Manga Text Cleaner — OCR + Inpainting          ║
║  manga-ocr  ·  LaMa Inpainting  ·  discord.py 2.x       ║
╚══════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, Awaitable

import discord
from discord import app_commands
from discord.ext import commands

# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("manga-bot")

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────
DISCORD_TOKEN     = os.getenv("DISCORD_TOKEN", "YOUR_TOKEN_HERE")
MAX_BYTES         = 50 * 1024 * 1024   # 50 MB
MAX_IMAGES        = 200
CONCURRENCY       = 3                  # صور تُعالج في نفس الوقت
THREAD_WORKERS    = 4                  # threads للـ CPU-bound (OCR + inpaint)
LONG_IMAGE_HEIGHT = 1400               # حد الصور الطويلة (مانهوا)
SLICE_HEIGHT      = 1200               # ارتفاع كل شريحة
SLICE_OVERLAP     = 100                # تداخل بين الشرائح
SUPPORTED_EXT     = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}

# ─────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────
_stats:       dict[int, dict]               = {}
_active_jobs: dict[int, asyncio.Event]      = {}
_executor:    ThreadPoolExecutor | None     = None
_ocr_model:   object | None                = None   # MangaOcr instance
_lama_model:  object | None                = None   # SimpleLama instance


# ─────────────────────────────────────────────────────────
# Model Loading  (lazy, thread-safe)
# ─────────────────────────────────────────────────────────
_ocr_lock  = asyncio.Lock()   # قفل مستقل لـ manga-ocr
_lama_lock = asyncio.Lock()   # قفل مستقل لـ LaMa → يتحملان بالتوازي

async def get_ocr_model():
    """يحمّل manga-ocr مرة واحدة فقط عند أول استخدام."""
    global _ocr_model
    # ✅ إصلاح 1: _executor قد يكون None لو استُدعي قبل on_ready
    if _executor is None:
        raise RuntimeError("البوت لم يكتمل تشغيله بعد، انتظر لحظة وأعد المحاولة.")
    async with _ocr_lock:
        if _ocr_model is None:
            log.info("⏳ تحميل manga-ocr …")
            loop = asyncio.get_running_loop()
            def _load():
                from manga_ocr import MangaOcr          # type: ignore
                return MangaOcr()
            _ocr_model = await loop.run_in_executor(_executor, _load)
            log.info("✅ manga-ocr جاهز")
    return _ocr_model

async def get_lama_model():
    """يحمّل LaMa inpainting مرة واحدة فقط."""
    global _lama_model
    if _executor is None:
        raise RuntimeError("البوت لم يكتمل تشغيله بعد، انتظر لحظة وأعد المحاولة.")
    async with _lama_lock:
        if _lama_model is None:
            log.info("⏳ تحميل LaMa inpainting …")
            loop = asyncio.get_running_loop()
            def _load():
                import torch
                from simple_lama_inpainting import SimpleLama  # type: ignore
                # ✅ إصلاح: تحميل النموذج على CPU صراحةً بغض النظر عن كيفية حفظه
                original_load = torch.jit.load
                def patched_load(f, *args, **kwargs):
                    kwargs.setdefault("map_location", torch.device("cpu"))
                    return original_load(f, *args, **kwargs)
                torch.jit.load = patched_load
                try:
                    model = SimpleLama()
                finally:
                    torch.jit.load = original_load  # استعادة الأصلي
                return model
            _lama_model = await loop.run_in_executor(_executor, _load)
            log.info("✅ LaMa جاهز")
    return _lama_model


# ─────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────
def stats_add(uid: int, images: int = 0, zips: int = 0, errors: int = 0) -> None:
    if uid not in _stats:
        _stats[uid] = {"images": 0, "zips": 0, "errors": 0, "last": None}
    _stats[uid]["images"] += images
    _stats[uid]["zips"]   += zips
    _stats[uid]["errors"] += errors
    _stats[uid]["last"]    = datetime.now()


# ─────────────────────────────────────────────────────────
# Job helpers
# ─────────────────────────────────────────────────────────
def job_register(uid: int) -> asyncio.Event:
    ev = asyncio.Event()
    _active_jobs[uid] = ev
    return ev

def job_finish(uid: int) -> None:
    _active_jobs.pop(uid, None)

def job_cancelled(uid: int) -> bool:
    ev = _active_jobs.get(uid)
    return ev is not None and ev.is_set()


# ─────────────────────────────────────────────────────────
# Natural sort
# ─────────────────────────────────────────────────────────
def _natural_key(path: str) -> list:
    parts = re.split(r"(\d+)", Path(path).name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# ─────────────────────────────────────────────────────────
# Core: OCR bubble detection
# ─────────────────────────────────────────────────────────
def _detect_bubbles_cv2(img_rgb) -> list[tuple[int,int,int,int]]:
    """
    يكتشف بالونات الكلام بـ OpenCV:
    1. Threshold → contours
    2. فلتر الأشكال الدائرية/البيضاوية الكبيرة
    يرجع قائمة (x, y, w, h)
    """
    import cv2
    import numpy as np

    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w  = gray.shape

    # Adaptive threshold لاكتشاف المناطق البيضاء (البالونات)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15, C=2
    )

    # morphology لإزالة الضوضاء
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int,int,int,int]] = []
    min_area = (h * w) * 0.002   # 0.2% من مساحة الصورة

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # نسبة العرض/الارتفاع — البالونات عادةً شكلها قريب من الدائرة/البيضاوية
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if aspect < 0.2 or aspect > 5.0:
            continue

        # تأكد إن الـ bounding box مش كل الصورة
        if bw > w * 0.9 or bh > h * 0.9:
            continue

        # padding صغير
        pad = 6
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes


def _ocr_filter_boxes(
    img_pil,
    boxes: list[tuple[int,int,int,int]],
    ocr,
) -> list[tuple[int,int,int,int]]:
    """
    يمرر كل box على manga-ocr:
    - لو رجع نص → هي بالونة كلام حقيقية
    - لو فاضي → تُتجاهل
    """
    confirmed: list[tuple[int,int,int,int]] = []
    for (x, y, w, h) in boxes:
        if w < 10 or h < 10:
            continue
        crop = img_pil.crop((x, y, x + w, y + h))
        try:
            text = ocr(crop)
            if text and text.strip():
                confirmed.append((x, y, w, h))
        except Exception:
            pass   # box مش نص — تخطي
    return confirmed


def _lama_inpaint(
    img_pil,
    boxes: list[tuple[int,int,int,int]],
    lama,
) -> object:  # PIL.Image
    """
    يبني mask من الـ boxes ويطبق LaMa inpainting.
    """
    # ✅ إصلاح 3: دمج الـ imports في مكان واحد بدل استيراد مكررين منفصلين
    from PIL import Image, ImageDraw

    w, h = img_pil.size
    mask = Image.new("L", (w, h), 0)

    draw = ImageDraw.Draw(mask)
    for (x, y, bw, bh) in boxes:
        draw.rectangle([x, y, x + bw, y + bh], fill=255)

    # LaMa يحتاج RGB
    result = lama(img_pil.convert("RGB"), mask)
    return result


# ─────────────────────────────────────────────────────────
# Core: process single image (sync — runs in executor)
# ─────────────────────────────────────────────────────────
def _process_image_sync(
    img_bytes: bytes,
    mode: str,
    ocr,
    lama,
) -> bytes:
    """
    Pipeline كامل لصورة واحدة:
    1. كشف البالونات بـ OpenCV
    2. تأكيد النص بـ manga-ocr
    3. مسح + ترميم بـ LaMa
    يشتغل في thread منفصل (CPU-bound).
    """
    import numpy as np
    from PIL import Image

    img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_rgb   = np.array(img_pil)

    # 1. كشف البالونات
    boxes = _detect_bubbles_cv2(img_rgb)
    if not boxes:
        return img_bytes   # مفيش بالونات — رجع الأصل

    # 2. تأكيد بـ OCR
    confirmed = _ocr_filter_boxes(img_pil, boxes, ocr)
    if not confirmed:
        return img_bytes   # مفيش نص فعلي

    log.info(f"    {len(confirmed)}/{len(boxes)} بالونة فيها نص")

    # 3. LaMa inpainting
    result_pil = _lama_inpaint(img_pil, confirmed, lama)

    buf = io.BytesIO()
    result_pil.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────
# Core: process long image (manhwa — slice & stitch)
# ─────────────────────────────────────────────────────────
def _process_long_image_sync(
    img_bytes: bytes,
    mode: str,
    ocr,
    lama,
) -> bytes:
    import numpy as np
    from PIL import Image

    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H    = img_pil.size

    if H <= SLICE_HEIGHT:
        return _process_image_sync(img_bytes, mode, ocr, lama)

    # بناء قائمة الشرائح
    slices: list[tuple[int,int]] = []
    y = 0
    while y < H:
        y_end = min(y + SLICE_HEIGHT, H)
        slices.append((y, y_end))
        if y_end >= H:
            break
        y = y_end - SLICE_OVERLAP

    result_arr = np.array(img_pil)

    for idx, (y0, y1) in enumerate(slices):
        crop     = img_pil.crop((0, y0, W, y1))
        crop_buf = io.BytesIO()
        crop.save(crop_buf, format="PNG")
        crop_bytes = crop_buf.getvalue()

        try:
            cleaned_bytes = _process_image_sync(crop_bytes, mode, ocr, lama)
            # ✅ إصلاح: إغلاق الصورة فور الانتهاء لتحرير الذاكرة
            with Image.open(io.BytesIO(cleaned_bytes)).convert("RGB") as _ci:
                cleaned_arr = np.array(_ci)

            write_y      = y0 if idx == 0 else y0 + SLICE_OVERLAP
            src_offset   = 0  if idx == 0 else SLICE_OVERLAP
            available    = cleaned_arr.shape[0] - src_offset
            copy_h       = min(y1 - write_y, available)

            if copy_h > 0:
                result_arr[write_y : write_y + copy_h] = (
                    cleaned_arr[src_offset : src_offset + copy_h]
                )
        except Exception as e:
            log.warning(f"  شريحة {idx+1} فشلت: {e} — تُتخطى")

    out = Image.fromarray(result_arr)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────
# Core: process ZIP (async + concurrent)
# ─────────────────────────────────────────────────────────
async def process_zip(
    zip_bytes: bytes,
    mode: str,
    uid: int,
    on_progress: Callable[[int, int, str], Awaitable[None]],
    ocr,
    lama,
) -> tuple[bytes, int, int]:
    """
    يعالج ZIP ويرجع (zip_bytes_new, success, total).
    يشغّل الصور بشكل متوازي بـ Semaphore.
    """
    # ── تحقق من ZIP ──
    buf = io.BytesIO(zip_bytes)
    if not zipfile.is_zipfile(buf):
        raise ValueError("الملف ليس ZIP صالح!")
    buf.seek(0)

    with zipfile.ZipFile(buf, "r") as zf:
        bad = zf.testzip()
        if bad:
            raise ValueError(f"ملف تالف: {bad}")

        image_files = sorted(
            [
                n for n in zf.namelist()
                if Path(n).suffix.lower() in SUPPORTED_EXT
                and "__MACOSX" not in n
                and not Path(n).name.startswith(".")
            ],
            key=_natural_key,
        )

        if not image_files:
            raise ValueError("لا توجد صور داخل الـ ZIP!")
        if len(image_files) > MAX_IMAGES:
            raise ValueError(
                f"الـ ZIP يحتوي {len(image_files)} صورة، الحد الأقصى {MAX_IMAGES}."
            )

        raw: dict[str, bytes] = {}
        for name in image_files:
            try:
                raw[name] = zf.read(name)
            except Exception as e:
                log.warning(f"تخطي '{name}': {e}")
    # ← zf مُغلق بأمان

    if not raw:
        raise ValueError("كل الصور في ZIP تالفة أو لا يمكن قراءتها!")

    total        = len(raw)
    success      = 0
    errors       = 0
    cleaned      : dict[str, bytes] = {}
    done_count   = 0
    lock         = asyncio.Lock()
    sem          = asyncio.Semaphore(CONCURRENCY)
    loop         = asyncio.get_running_loop()
    names        = list(raw.keys())

    async def process_one(name: str, img_bytes: bytes) -> None:
        nonlocal success, errors, done_count

        async with sem:
            if job_cancelled(uid):
                raise asyncio.CancelledError()

            try:
                from PIL import Image
                # ✅ إصلاح 2: استخدام context manager لإغلاق الصورة وتحرير الذاكرة
                with Image.open(io.BytesIO(img_bytes)) as _img:
                    img_w, img_h = _img.size
                is_long  = mode in ("manhwa", "manhua") and img_h > LONG_IMAGE_HEIGHT

                fn = _process_long_image_sync if is_long else _process_image_sync
                result = await loop.run_in_executor(
                    _executor, fn, img_bytes, mode, ocr, lama
                )

                async with lock:
                    cleaned[name]  = result
                    success       += 1
                    done_count    += 1
                    await on_progress(done_count, total, Path(name).name)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"❌ {name}: {e}")
                async with lock:
                    cleaned[name]  = img_bytes
                    errors        += 1
                    done_count    += 1
                    await on_progress(done_count, total, Path(name).name)

    tasks = [asyncio.create_task(process_one(n, raw[n])) for n in names]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    # ── بناء ZIP الناتج ──
    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in image_files:
            if name in cleaned:
                zout.writestr(name, cleaned[name])
    # ← zout مُغلق، flush تم

    out_buf.seek(0)
    stats_add(uid, images=success, zips=1, errors=errors)
    return out_buf.read(), success, total


# ─────────────────────────────────────────────────────────
# Discord Bot
# ─────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


# ── Embed builders ────────────────────────────────────────
def _embed_result(name: str, success: int, total: int, mode: str) -> discord.Embed:
    label = {"manga": "مانجا 🇯🇵", "manhwa": "مانهوا 🇰🇷", "manhua": "مانهوا 🇨🇳"}.get(mode, mode)
    color = 0x57F287 if success == total else 0xFEE75C
    e = discord.Embed(title="✅ تم تنظيف الصور!", color=color)
    e.add_field(name="📦 الملف",    value=f"`{name}`",               inline=True)
    e.add_field(name="🎯 النوع",    value=label,                      inline=True)
    e.add_field(name="📊 النتيجة", value=f"`{success}/{total}` صورة", inline=True)
    e.add_field(name="🔬 OCR",      value="`manga-ocr`",              inline=True)
    e.add_field(name="🎨 Inpaint",  value="`LaMa`",                   inline=True)
    if success < total:
        e.add_field(
            name="⚠️ ملاحظة",
            value=f"{total - success} صورة فشلت — احتُفظ بنسختها الأصلية.",
            inline=False,
        )
    return e


# ── Core handler (shared by slash + message) ─────────────
async def handle_zip(
    attachment: discord.Attachment,
    mode: str,
    uid: int,
    channel: discord.abc.Messageable,
    reply: Callable,
    edit: Callable,
) -> None:

    if not attachment.filename.lower().endswith(".zip"):
        await reply(content="❌ ارفع ملف `.zip` فقط!")
        return

    if attachment.size > MAX_BYTES:
        await reply(content=f"❌ الملف `{attachment.size/1024/1024:.1f} MB` أكبر من الحد `{MAX_BYTES//1024//1024} MB`.")
        return

    if uid in _active_jobs:
        await reply(content="⚠️ عندك عملية شغالة! استخدم `/cancel` لإلغائها.")
        return

    status = await reply(content="📥 جاري تحميل الملف…")

    try:
        zip_bytes = await attachment.read()
    except Exception as e:
        await edit(status, content=f"❌ فشل تحميل الملف: `{e}`")
        return

    try:
        await edit(status, content="🔬 تحميل نماذج OCR + LaMa… (قد يأخذ دقيقتين في أول مرة)")
    except discord.HTTPException:
        pass

    # ✅ تحديث الرسالة كل 10 ثوانٍ أثناء تحميل النماذج لمنع ظهور "عالق"
    loading_task = None
    async def _keep_alive():
        dots = 0
        while True:
            await asyncio.sleep(10)
            dots = (dots % 3) + 1
            try:
                await edit(status, content=f"🔬 تحميل نماذج OCR + LaMa{'.' * dots} (قد يأخذ دقيقتين في أول مرة)")
            except discord.HTTPException:
                pass

    loading_task = asyncio.create_task(_keep_alive())
    try:
        # ✅ تحميل تسلسلي لتقليل ذروة الذاكرة (OCR أولاً ثم LaMa)
        ocr_ready  = await asyncio.wait_for(get_ocr_model(), timeout=300)
        lama_ready = await asyncio.wait_for(get_lama_model(), timeout=300)
    except asyncio.TimeoutError:
        loading_task.cancel()
        try:    await edit(status, content="❌ انتهت مهلة تحميل النماذج (5 دقائق). أعد المحاولة لاحقاً.")
        except discord.HTTPException:
            await channel.send("❌ انتهت مهلة تحميل النماذج. أعد المحاولة.")
        return
    except Exception as ex:
        loading_task.cancel()
        try:    await edit(status, content=f"❌ فشل تحميل النماذج: `{ex}`")
        except discord.HTTPException:
            await channel.send(f"❌ فشل تحميل النماذج: `{ex}`")
        return
    finally:
        if loading_task and not loading_task.done():
            loading_task.cancel()

    try:
        await edit(status, content="✅ النماذج جاهزة — بدء تنظيف الصور…")
    except discord.HTTPException:
        pass

    # ✅ إصلاح 1: job_finish مضمون حتى عند الـ return المبكر بـ try/finally صريح
    job_register(uid)
    try:
        async def on_progress(done: int, total: int, name: str) -> None:
            if total == 0:
                return
            pct = int(done / total * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            try:
                await edit(
                    status,
                    content=(
                        f"⚙️ **جاري تنظيف الصور…** _(اكتب `/cancel` للإلغاء)_\n"
                        f"`[{bar}]` {pct}%  ·  `{done}/{total}` — {name}"
                    ),
                )
            except discord.HTTPException:
                pass

        try:
            new_zip, success, total = await process_zip(zip_bytes, mode, uid, on_progress, ocr_ready, lama_ready)

        except asyncio.CancelledError:
            try:    await edit(status, content="🚫 **تم إلغاء العملية.**")
            except discord.HTTPException:
                await channel.send("🚫 **تم إلغاء العملية.**")
            return

        except ValueError as e:
            msg = f"❌ {e}"[:1900]
            try:    await edit(status, content=msg)
            except discord.HTTPException:
                await channel.send(msg)
            return

        except Exception as e:
            log.exception("خطأ غير متوقع في process_zip")
            # ✅ إصلاح: تقليص رسالة الخطأ — PyTorch يرسل رسائل أطول من 2000 حرف
            short_err = str(e).split("\n")[0][:300]
            msg = f"❌ خطأ غير متوقع: `{short_err}`"
            try:    await edit(status, content=msg)
            except discord.HTTPException:
                await channel.send(msg)
            return

        out_name = Path(attachment.filename).stem + "_cleaned.zip"
        embed    = _embed_result(out_name, success, total, mode)

        try:    await edit(status, content="", embed=embed)
        except discord.HTTPException:
            await channel.send(embed=embed)

        # ✅ إصلاح: إرسال الملف عبر reply() للـ slash commands
        # channel.send() يفشل بعد انتهاء الـ interaction timeout
        try:
            await reply(
                file=discord.File(io.BytesIO(new_zip), filename=out_name)
            )
        except discord.HTTPException as e:
            msg = (
                "❌ الملف الناتج أكبر من حد Discord (25MB). جرب ZIP أصغر."
                if e.status == 413
                else f"❌ فشل إرسال الملف: `{e.text}`"
            )
            await channel.send(msg)

    finally:
        job_finish(uid)   # ✅ يُنفَّذ دايماً بغض النظر عن أي return أو exception


# ══════════════════════════════════════════════════════════
# Slash Commands
# ══════════════════════════════════════════════════════════

@bot.tree.command(name="clean", description="احذف نصوص المانجا/المانهوا من ZIP بـ OCR + LaMa")
@app_commands.describe(file="ملف ZIP يحتوي على صفحات المانجا أو المانهوا", mode="نوع المحتوى")
@app_commands.choices(mode=[
    app_commands.Choice(name="🇯🇵 مانجا (يابانية)",         value="manga"),
    app_commands.Choice(name="🇰🇷 مانهوا (كورية - طويلة)", value="manhwa"),
    app_commands.Choice(name="🇨🇳 مانهوا (صينية - طويلة)", value="manhua"),
])
async def cmd_clean(
    interaction: discord.Interaction,
    file: discord.Attachment,
    mode: app_commands.Choice[str] | None = None,
) -> None:
    await interaction.response.defer(thinking=True)

    async def reply(content=None, **kw):
        return await interaction.followup.send(content=content, **kw)
    async def edit(msg, content=None, **kw):
        await msg.edit(content=content, **kw)

    await handle_zip(
        file, mode.value if mode else "manga",
        interaction.user.id, interaction.channel,
        reply, edit,
    )


@bot.tree.command(name="cancel", description="إلغاء عملية التنظيف الشغالة")
async def cmd_cancel(interaction: discord.Interaction) -> None:
    uid = interaction.user.id
    if uid not in _active_jobs:
        await interaction.response.send_message("ℹ️ مفيش عملية شغالة.", ephemeral=True)
        return
    _active_jobs[uid].set()
    await interaction.response.send_message("🚫 **جاري الإيقاف…**", ephemeral=True)


@bot.tree.command(name="stats", description="إحصائياتك مع البوت")
async def cmd_stats(interaction: discord.Interaction) -> None:
    uid  = interaction.user.id
    data = _stats.get(uid)

    e = discord.Embed(title="📊 إحصائياتك", color=0x5865F2)
    e.set_author(name=interaction.user.display_name, icon_url=interaction.user.display_avatar.url)

    if not data:
        e.description = "لم تستخدم البوت بعد! جرّب `/clean` 🗒️"
    else:
        last = data["last"].strftime("%Y-%m-%d %H:%M") if data["last"] else "—"
        e.add_field(name="🖼️ صور نُظِّفت", value=f"`{data['images']}`", inline=True)
        e.add_field(name="📦 ZIP معالجة",   value=f"`{data['zips']}`",   inline=True)
        e.add_field(name="❌ أخطاء",         value=f"`{data['errors']}`", inline=True)
        e.add_field(name="🕐 آخر استخدام",  value=f"`{last}`",           inline=False)

    if uid in _active_jobs:
        e.add_field(name="⚙️ الحالة", value="عملية شغالة — `/cancel` للإلغاء", inline=False)

    e.set_footer(text="⚠️ الإحصائيات تُمسح عند إعادة تشغيل البوت")
    await interaction.response.send_message(embed=e, ephemeral=True)


@bot.tree.command(name="help", description="تعليمات البوت")
async def cmd_help(interaction: discord.Interaction) -> None:
    e = discord.Embed(
        title="🗒️ Manga Text Cleaner",
        description="يحذف نصوص المانجا والمانهوا بدقة عالية باستخدام:\n**manga-ocr** للتعرف على النص · **LaMa** لترميم الخلفية",
        color=0x5865F2,
    )
    e.add_field(
        name="📋 الأوامر",
        value=(
            "`/clean`  — ارفع ZIP واختار النوع\n"
            "`/cancel` — إلغاء عملية شغالة\n"
            "`/stats`  — إحصائياتك\n"
            "`/help`   — هذه الرسالة"
        ),
        inline=False,
    )
    e.add_field(name="🎯 الأنواع",  value="🇯🇵 مانجا · 🇰🇷 مانهوا · 🇨🇳 مانهوا صيني", inline=True)
    e.add_field(name="📁 الصيغ",   value="jpg  png  webp  bmp  tiff  gif",              inline=True)
    e.add_field(name="📦 حد ZIP",  value=f"`{MAX_BYTES//1024//1024} MB / {MAX_IMAGES} صورة`", inline=True)
    e.add_field(
        name="⚡ اكتشاف تلقائي",
        value='اسم الملف يحتوي "manhwa" أو "manhua" → يُعرَف تلقائياً',
        inline=False,
    )
    e.set_footer(text="manga-ocr + LaMa Inpainting")
    await interaction.response.send_message(embed=e)


# ─────────────────────────────────────────────────────────
# on_message  (drag & drop)
# ─────────────────────────────────────────────────────────
@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return

    for att in message.attachments:
        if att.filename.lower().endswith(".zip"):
            fname = att.filename.lower()
            if "manhwa" in fname or "webtoon" in fname:
                mode = "manhwa"
            elif "manhua" in fname:
                mode = "manhua"
            else:
                mode = "manga"

            # ✅ إصلاح: تمرير كل الـ kwargs (يشمل file=) لدعم إرسال الملف
            async def reply(content=None, _m=message, **kw):
                return await _m.reply(content=content, **kw)
            async def edit(msg, content=None, _att=att, **kw):
                await msg.edit(content=content, **kw)

            await handle_zip(att, mode, message.author.id, message.channel, reply, edit)
            break

    await bot.process_commands(message)


# ─────────────────────────────────────────────────────────
# on_ready
# ─────────────────────────────────────────────────────────
@bot.event
async def on_ready() -> None:
    global _executor, _ocr_model, _lama_model
    # ✅ إصلاح: عند إعادة الاتصال — أغلق الـ executor القديم وأعد تهيئة النماذج
    if _executor is not None:
        _executor.shutdown(wait=False)
        _ocr_model  = None   # النماذج ستُعاد بالـ executor الجديد
        _lama_model = None
    _executor = ThreadPoolExecutor(max_workers=THREAD_WORKERS, thread_name_prefix="manga")
    log.info(f"🤖 {bot.user}  (ID: {bot.user.id})")

    try:
        synced = await bot.tree.sync()
        log.info(f"✅ {len(synced)} slash commands synced")
    except Exception as e:
        log.error(f"❌ فشل sync: {e}")

    await bot.change_presence(
        activity=discord.Activity(type=discord.ActivityType.watching, name="ZIP files | /clean")
    )

    # Pre-warm النماذج في الخلفية بالتوازي (يسرّع أول طلب)
    async def _warm():
        try:
            # تحميل تسلسلي في الـ pre-warm لتقليل ذروة الذاكرة
            await get_ocr_model()
            await get_lama_model()
            log.info("🔥 النماذج جاهزة في الذاكرة")
        except Exception as e:
            log.warning(f"⚠️ Pre-warm فشل (سيُعاد التحميل عند أول طلب): {e}")

    task = asyncio.create_task(_warm())
    # ✅ إصلاح 4: منع الـ task من الاختفاء بصمت لو طلع exception
    task.add_done_callback(
        lambda t: log.error(f"❌ _warm انتهى بخطأ: {t.exception()}") if not t.cancelled() and t.exception() else None
    )


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    if DISCORD_TOKEN == "YOUR_TOKEN_HERE":
        log.error("❌ ضع DISCORD_TOKEN في متغيرات البيئة أو في الكود!")
        sys.exit(1)   # ✅ إصلاح: خروج بكود خطأ بدل الصمت
    bot.run(DISCORD_TOKEN, log_handler=None)
