import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import fitz
from fontTools.ttLib import TTFont as FT_Font
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


# -------------------------------------------------------------
# 1. NHẬP FONT VÀO PDF
# -------------------------------------------------------------
def set_font(pdf, font_path, config):
    """Sử dụng font_name và font_size từ config mới."""
    pdfmetrics.registerFont(TTFont(config.sheet.font_name, font_path))
    pdf.setFont(config.sheet.font_name, config.sheet.font_size)


# -------------------------------------------------------------
# 2. TẠO ENGLISH HANDWRITING SHEET
# -------------------------------------------------------------
def create_english_sheet(font_path, config):
    # save_path trong config mới đã được get_path xử lý lúc khởi tạo
    path = Path(config.sheet.save_path) / config.sheet.en_filename

    # Chuyển Path object sang string cho reportlab
    pdf = canvas.Canvas(str(path), pagesize=A4)
    set_font(pdf, font_path, config)

    y = config.sheet.margin_top
    phan_tren = True

    # Lấy text từ config mới
    text_content = config.sheet.en_text.strip().splitlines()

    for line in text_content:
        mau = True
        while True:
            # Tính toán x dựa trên margins mới
            text_w = stringWidth(
                line, config.sheet.font_name, config.sheet.font_size
            )
            if text_w < (config.sheet.margin_right - 40):
                x = config.sheet.margin_left + (
                    (
                        config.sheet.margin_right
                        - config.sheet.margin_left
                        - text_w
                    )
                    / 2
                )
            else:
                x = config.sheet.margin_left

            # Vẽ đường kẻ mờ
            offset = config.sheet.font_size * 0.1
            pdf.setStrokeColorRGB(0.9, 0.9, 0.9)
            pdf.line(10, y - offset, 584, y - offset)  # A4 width ~595

            # Đổi màu mực
            if mau:
                pdf.setFillColorRGB(0, 0, 0)
                mau = False
            else:
                pdf.setFillColorRGB(0.6, 0.6, 0.6)

            pdf.drawString(x, y, line)
            y -= config.sheet.line_spacing

            # Sử dụng các mốc chia lưới mới
            if y < config.sheet.divide_horizontal and phan_tren:
                phan_tren = False
                break
            if y < 20:
                break

    pdf.save()
    _convert_pdf_to_png(str(path))


# -------------------------------------------------------------
# 3. TẠO VIETNAMESE HANDWRITING SHEET
# -------------------------------------------------------------
def create_vietnamese_sheet(font_path, config):
    path = Path(config.sheet.save_path) / config.sheet.vi_filename

    pdf = canvas.Canvas(str(path), pagesize=A4)
    set_font(pdf, font_path, config)

    phan_trai = True
    x_pos = config.sheet.margin_left
    y_pos = config.sheet.margin_top

    for vi_char in config.sheet.vi_text.split():
        chu_mau = True
        x = x_pos
        # divide_vertical thay cho chia_doc
        gioi_han_phai = (
            config.sheet.divide_vertical
            if phan_trai
            else config.sheet.margin_right
        )

        offset = config.sheet.font_size * 0.1
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
            x += config.sheet.word_spacing

            if x > (gioi_han_phai - 10):
                break

        # Kẻ đường dọc chia trang
        pdf.setStrokeColorRGB(0.2, 0.2, 0.2)
        pdf.line(
            config.sheet.divide_vertical,
            830,
            config.sheet.divide_vertical,
            10,
        )

        y_pos -= config.sheet.line_spacing

        if y_pos < config.sheet.margin_bottom:
            if phan_trai:
                x_pos = config.sheet.divide_vertical + 10
                y_pos = config.sheet.margin_top
                phan_trai = False
            else:
                phan_trai = True
                x_pos = config.sheet.margin_left
                y_pos = config.sheet.margin_top
                pdf.showPage()
                set_font(pdf, font_path, config)

    pdf.save()
    _convert_pdf_to_png(str(path))


# -------------------------------------------------------------
# 4. HÀM PHỤ TRỢ (PNG CONVERSION & FONT CHECK)
# -------------------------------------------------------------
def _convert_pdf_to_png(pdf_path: str):
    """Hàm nội bộ để chuyển PDF sang ảnh PNG."""
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            image_name = pdf_path.replace(
                ".pdf", f"_page_{page_num + 1}.png"
            )
            pix.save(image_name)
        doc.close()
    except Exception as e:
        print(f"Lỗi khi tạo ảnh từ PDF: {e}")


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
    root = tk.Tk()
    root.withdraw()
    font_path = filedialog.askopenfilename(
        title="Hãy chọn một file Font chữ",
        filetypes=[("Font TrueType", "*.ttf"), ("Tất cả", "*.*")],
    )
    return font_path if font_path else None


def create_handwriting_sheet(config):
    """Hàm chính: Chỉ cần truyền object config vào."""
    font_path = input_font()
    if not font_path:
        print("Hủy thao tác chọn font.")
        return

    # Tự động chọn sheet dựa trên font
    if is_vietnamese_font(font_path, config):
        print("Đang tạo Vietnamese Sheet...")
        create_vietnamese_sheet(font_path, config)
    else:
        print("Đang tạo English Sheet...")
        create_english_sheet(font_path, config)
