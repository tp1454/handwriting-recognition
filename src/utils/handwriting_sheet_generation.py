from __future__ import annotations

import copy
import re
from datetime import datetime
from pathlib import Path

try:
    import fitz as _fitz
except Exception:  # pragma: no cover - environment dependent
    _fitz = None
from fontTools.ttLib import TTFont as FT_Font
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

_ALLOWED_LANGUAGES = {"auto", "en", "vi"}
_POSITIVE_FIELDS = {
    "font_size",
    "line_spacing",
    "word_spacing",
}
_NON_NEGATIVE_FIELDS = {
    "margin_left",
    "margin_right",
    "margin_top",
    "margin_bottom",
    "divide_horizontal",
    "divide_vertical",
}
_BOOLEAN_FIELDS = {"show_vertical_line"}


def _get_fitz_module():
    if _fitz is not None:
        return _fitz

    try:
        import pymupdf as pymupdf_fitz

        return pymupdf_fitz
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyMuPDF is unavailable. Install pymupdf."
        ) from exc


# -------------------------------------------------------------
# 1. NHẬP FONT VÀO PDF
# -------------------------------------------------------------
def _build_font_alias(base_name: str, font_path: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_]+", "_", Path(font_path).stem)
    suffix = suffix.strip("_") or "font"
    return f"{base_name}_{suffix}"[:80]


def set_font(pdf, font_path, sheet_config):
    """Register font safely and return the font name used by reportlab."""
    font_name = _build_font_alias(
        str(sheet_config.font_name), font_path
    )
    try:
        pdfmetrics.getFont(font_name)
    except KeyError:
        pdfmetrics.registerFont(TTFont(font_name, font_path))

    pdf.setFont(font_name, sheet_config.font_size)
    return font_name


def _coerce_sheet_value(field: str, value: int | float | str) -> int:
    numeric = int(value)
    if field in _POSITIVE_FIELDS and numeric <= 0:
        raise ValueError(f"{field} must be > 0")
    if field in _NON_NEGATIVE_FIELDS and numeric < 0:
        raise ValueError(f"{field} must be >= 0")
    return numeric


def _coerce_sheet_bool(field: str, value: bool | str | int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{field} must be a boolean")


def _build_sheet_settings(
    config,
    *,
    language: str,
    output_filename: str,
    custom_text: str | None = None,
    overrides: dict[str, int | bool] | None = None,
):
    sheet = copy.deepcopy(config.sheet)

    if overrides:
        for field, raw_value in overrides.items():
            if raw_value is None:
                continue
            if (
                field not in _POSITIVE_FIELDS
                and field not in _NON_NEGATIVE_FIELDS
                and field not in _BOOLEAN_FIELDS
            ):
                continue
            if field in _BOOLEAN_FIELDS:
                setattr(
                    sheet,
                    field,
                    _coerce_sheet_bool(field, raw_value),
                )
            else:
                setattr(
                    sheet,
                    field,
                    _coerce_sheet_value(field, raw_value),
                )

    if custom_text and custom_text.strip():
        if language == "vi":
            sheet.vi_text = custom_text.strip()
        else:
            sheet.en_text = custom_text.strip()

    if language == "vi":
        sheet.vi_filename = output_filename
    else:
        sheet.en_filename = output_filename

    return sheet


# -------------------------------------------------------------
# 2. TẠO ENGLISH HANDWRITING SHEET
# -------------------------------------------------------------
def create_english_sheet(
    font_path,
    config,
    output_filename: str,
    custom_text: str | None = None,
    overrides: dict[str, int | bool] | None = None,
):
    sheet = _build_sheet_settings(
        config,
        language="en",
        output_filename=output_filename,
        custom_text=custom_text,
        overrides=overrides,
    )

    # save_path trong config mới đã được get_path xử lý lúc khởi tạo
    path = Path(sheet.save_path) / sheet.en_filename

    # Chuyển Path object sang string cho reportlab
    pdf = canvas.Canvas(str(path), pagesize=A4)
    font_name = set_font(pdf, font_path, sheet)

    y = sheet.margin_top
    phan_tren = True

    # Lấy text từ config mới
    text_content = sheet.en_text.strip().splitlines()

    for line in text_content:
        mau = True
        while True:
            # Tính toán x dựa trên margins mới
            text_w = stringWidth(line, font_name, sheet.font_size)
            if text_w < (sheet.margin_right - 40):
                x = sheet.margin_left + (
                    (sheet.margin_right - sheet.margin_left - text_w)
                    / 2
                )
            else:
                x = sheet.margin_left

            # Vẽ đường kẻ mờ
            offset = sheet.font_size * 0.1
            pdf.setStrokeColorRGB(0.9, 0.9, 0.9)
            pdf.line(10, y - offset, 584, y - offset)  # A4 width ~595

            # Đổi màu mực
            if mau:
                pdf.setFillColorRGB(0, 0, 0)
                mau = False
            else:
                pdf.setFillColorRGB(0.6, 0.6, 0.6)

            pdf.drawString(x, y, line)
            y -= sheet.line_spacing

            # Sử dụng các mốc chia lưới mới
            if y < sheet.divide_horizontal and phan_tren:
                phan_tren = False
                break
            if y < 20:
                break

    pdf.save()
    png_paths = _convert_pdf_to_png(str(path))
    return {
        "pdf_path": str(path),
        "preview_png_path": png_paths[0] if png_paths else "",
        "png_paths": png_paths,
    }


# -------------------------------------------------------------
# 3. TẠO VIETNAMESE HANDWRITING SHEET
# -------------------------------------------------------------
def create_vietnamese_sheet(
    font_path,
    config,
    output_filename: str,
    custom_text: str | None = None,
    overrides: dict[str, int | bool] | None = None,
):
    sheet = _build_sheet_settings(
        config,
        language="vi",
        output_filename=output_filename,
        custom_text=custom_text,
        overrides=overrides,
    )

    path = Path(sheet.save_path) / sheet.vi_filename

    pdf = canvas.Canvas(str(path), pagesize=A4)
    set_font(pdf, font_path, sheet)

    phan_trai = True
    x_pos = sheet.margin_left
    y_pos = sheet.margin_top

    for vi_char in sheet.vi_text.split():
        chu_mau = True
        x = x_pos
        # divide_vertical thay cho chia_doc
        gioi_han_phai = (
            sheet.divide_vertical if phan_trai else sheet.margin_right
        )

        offset = sheet.font_size * 0.1
        pdf.setStrokeColorRGB(0.9, 0.9, 0.9)
        pdf.line(
            x_pos - 5,
            y_pos - offset,
            gioi_han_phai - 5,
            y_pos - offset,
        )

        while True:
            (
                pdf.setFillColorRGB(0, 0, 0)
                if chu_mau
                else pdf.setFillColorRGB(0.7, 0.7, 0.7)
            )
            chu_mau = False

            pdf.drawString(x, y_pos, vi_char)
            x += sheet.word_spacing

            if x > (gioi_han_phai - 10):
                break

        if getattr(sheet, "show_vertical_line", False):
            # Kẻ đường dọc chia trang
            pdf.setStrokeColorRGB(0.2, 0.2, 0.2)
            pdf.line(
                sheet.divide_vertical,
                830,
                sheet.divide_vertical,
                10,
            )

        y_pos -= sheet.line_spacing

        if y_pos < sheet.margin_bottom:
            if phan_trai:
                x_pos = sheet.divide_vertical + 10
                y_pos = sheet.margin_top
                phan_trai = False
            else:
                phan_trai = True
                x_pos = sheet.margin_left
                y_pos = sheet.margin_top
                pdf.showPage()
                set_font(pdf, font_path, sheet)

    pdf.save()
    png_paths = _convert_pdf_to_png(str(path))
    return {
        "pdf_path": str(path),
        "preview_png_path": png_paths[0] if png_paths else "",
        "png_paths": png_paths,
    }


# -------------------------------------------------------------
# 4. HÀM PHỤ TRỢ (PNG CONVERSION & FONT CHECK)
# -------------------------------------------------------------
def _convert_pdf_to_png(pdf_path: str):
    """Hàm nội bộ để chuyển PDF sang ảnh PNG."""
    generated: list[str] = []
    try:
        fitz_module = _get_fitz_module()
        doc = fitz_module.open(pdf_path)
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            image_name = pdf_path.replace(
                ".pdf", f"_page_{page_num + 1}.png"
            )
            pix.save(image_name)
            generated.append(image_name)
        doc.close()
    except Exception as e:
        print(f"Lỗi khi tạo ảnh từ PDF: {e}")
    return generated


def is_vietnamese_font(font_path, config):
    try:
        bang_ma = FT_Font(font_path).getBestCmap()
        # Kiểm tra dựa trên bộ ký tự tiếng Việt trong config
        cac_chu_can_in = set(
            char for char in config.sheet.vi_text if char.strip()
        )

        for char in cac_chu_can_in:
            if ord(char) not in bang_ma:
                return False
        return True
    except Exception as e:
        print(f"Lỗi check font: {e}")
        return False


# -------------------------------------------------------------
# 5. GIAO DIỆN VÀ THỰC THI
# -------------------------------------------------------------
def input_font():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    font_path = filedialog.askopenfilename(
        title="Hãy chọn một file Font chữ",
        filetypes=[("Font TrueType", "*.ttf"), ("Tất cả", "*.*")],
    )
    return font_path if font_path else None


def _sanitize_output_basename(output_basename: str | None) -> str:
    if not output_basename:
        return ""
    cleaned = re.sub(
        r"[^A-Za-z0-9_-]+", "_", str(output_basename).strip()
    )
    return cleaned.strip("_")[:80]


def generate_handwriting_sheet(
    config,
    *,
    font_path: str,
    language: str = "auto",
    custom_text: str | None = None,
    overrides: dict[str, int | bool] | None = None,
    output_basename: str | None = None,
):
    resolved_font = Path(font_path).expanduser().resolve()
    if not resolved_font.exists():
        raise ValueError("Selected font file does not exist")

    normalized_language = str(language or "auto").strip().lower()
    if normalized_language not in _ALLOWED_LANGUAGES:
        raise ValueError("language must be one of auto, en, vi")

    final_language = normalized_language
    if final_language == "auto":
        final_language = (
            "vi"
            if is_vietnamese_font(str(resolved_font), config)
            else "en"
        )

    basename = _sanitize_output_basename(output_basename)
    if not basename:
        basename = (
            f"{final_language}_sheet_"
            f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        )
    output_filename = f"{basename}.pdf"

    if final_language == "vi":
        result = create_vietnamese_sheet(
            str(resolved_font),
            config,
            output_filename=output_filename,
            custom_text=custom_text,
            overrides=overrides,
        )
    else:
        result = create_english_sheet(
            str(resolved_font),
            config,
            output_filename=output_filename,
            custom_text=custom_text,
            overrides=overrides,
        )

    result["language"] = final_language
    return result


def create_handwriting_sheet(config):
    """Hàm chính: Chỉ cần truyền object config vào."""
    font_path = input_font()
    if not font_path:
        print("Hủy thao tác chọn font.")
        return

    result = generate_handwriting_sheet(
        config,
        font_path=font_path,
        language="auto",
    )
    if result.get("language") == "vi":
        print("Đang tạo Vietnamese Sheet...")
    else:
        print("Đang tạo English Sheet...")
    return result
