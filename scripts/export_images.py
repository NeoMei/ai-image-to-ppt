#!/usr/bin/env python3
"""通用图片打包工具: 图片列表 → PDF + PPTX (16:9, 统一尺寸, 工整一致).

特性:
  - 自动 normalize 所有图到 1920x1080 (LANCZOS + cover 裁剪, 不变形)
  - PDF: 12.8"×7.2" 页 (16:9, 150 DPI)
  - PPTX: 13.333"×7.5" slide (标准 16:9)

依赖:
  pip install Pillow python-pptx

用法:
  # CLI: 传入图片列表和输出前缀
  python3 export_images.py <output_prefix> <img1.jpg> <img2.jpg> ...

  # 脚本内调用
  from export_images import export_pdf, export_pptx
  files = ["slide1.jpg", "slide2.jpg", ...]
  export_pdf(files, "out.pdf")
  export_pptx(files, "out.pptx")
"""
import os, sys
from io import BytesIO
from PIL import Image
from pptx import Presentation
from pptx.util import Emu

TARGET_SIZE = (1920, 1080)  # 16:9 标准目标尺寸
SLIDE_W = Emu(int(13.333 * 914400))  # 13.333"
SLIDE_H = Emu(int(7.5 * 914400))     # 7.5"


def normalize(img):
    """将图片以 cover 模式缩放到 TARGET_SIZE (裁剪溢出, 保持比例)."""
    if img.size == TARGET_SIZE:
        return img
    tw, th = TARGET_SIZE
    w, h = img.size
    src_r = w / h
    tgt_r = tw / th
    if src_r > tgt_r:
        new_w = int(h * tgt_r)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    elif src_r < tgt_r:
        new_h = int(w / tgt_r)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    return img.resize(TARGET_SIZE, Image.LANCZOS)


def _load(path):
    img = normalize(Image.open(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def export_pdf(files, pdf_path):
    """图片列表 → 多页 PDF (统一 1920x1080)."""
    images = [_load(f) for f in files]
    images[0].save(
        pdf_path,
        save_all=True,
        append_images=images[1:],
        resolution=150.0,
        quality=85,
        optimize=True,
    )
    size_mb = os.path.getsize(pdf_path) / 1024 / 1024
    print(f"  PDF: {pdf_path} ({len(images)} 页, {size_mb:.1f}MB)")


def export_pptx(files, pptx_path):
    """图片列表 → PPTX (16:9 全屏, 不变形)."""
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]
    for f in files:
        img = _load(f)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        slide = prs.slides.add_slide(blank)
        slide.shapes.add_picture(buf, 0, 0, width=SLIDE_W, height=SLIDE_H)
    prs.save(pptx_path)
    size_mb = os.path.getsize(pptx_path) / 1024 / 1024
    print(f"  PPTX: {pptx_path} ({len(files)} slides, {size_mb:.1f}MB)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: export_images.py <output_prefix> <img1.jpg> <img2.jpg> ...")
        sys.exit(1)
    prefix = sys.argv[1]
    files = sys.argv[2:]
    export_pdf(files, f"{prefix}.pdf")
    export_pptx(files, f"{prefix}.pptx")
