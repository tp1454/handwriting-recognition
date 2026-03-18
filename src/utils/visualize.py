from .config import *
from pathlib import Path
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from fontTools.ttLib import TTFont as FT_Font


# I . hàm
# tìm địa chỉ FOLDER nhận OUTPUT
def tim_thu_muc(duong_dan):

    thu_muc_bat_dau = Path(__file__).resolve().parent
    cac_cap_thu_muc = [thu_muc_bat_dau] + list(thu_muc_bat_dau.parents)

    for thu_muc_cha in cac_cap_thu_muc:
        duong_dan_kiem_tra = thu_muc_cha / duong_dan

        if duong_dan_kiem_tra.is_dir():
            return str(duong_dan_kiem_tra)

    raise FileNotFoundError(
        f" Đã lùi hết cỡ nhưng không tìm thấy tổ hợp thư mục: '{duong_dan}'"
    )


# II . NHẬP FONT vào thư viện
def set_font(pdf, temp_font_path, font_size):
    # Đăng ký font với một tên duy nhất
    pdfmetrics.registerFont(TTFont(EN_FONT_NAME, temp_font_path))
    pdf.setFont(EN_FONT_NAME, font_size)


# III . TẠO SHEET HỖ TRỢ TIẾNG ANH :
def create_english_sheet(temp_font_path, duong_dan):
    # Khởi tạo pdf
    thu_muc = tim_thu_muc(duong_dan)
    duong_dan_luu_tep = os.path.join(thu_muc, EN_filename)
    pdf = canvas.Canvas(duong_dan_luu_tep, pagesize=A4)

    set_font(pdf, temp_font_path, size)  # nhập FONT

    y = le_tren
    phan_tren = True
    for EN_text in ENGLISH_TEXT.strip().splitlines():

        mau = True  # Biến để đổi màu dòng đầu tiên
        while True:

            # Căn giữa
            if stringWidth(EN_text, EN_FONT_NAME, size) < le_phai - 40:
                x = le_trai + (
                    (le_phai - le_trai - stringWidth(EN_text, EN_FONT_NAME, size)) / 2
                )
            else:
                x = le_trai

            if mau:
                pdf.setFillColorRGB(0, 0, 0)
                mau = False
            else:
                pdf.setFillColorRGB(0.6, 0.6, 0.6)

            # tạo hàng
            pdf.setDash(1, 3)
            pdf.setStrokeColorRGB(0.8, 0.8, 0.8)
            pdf.line(0, y, 595, y)
            pdf.setDash()

            pdf.drawString(x, y, EN_text)
            y -= size + 5

            # Điều kiện dừng
            if y < chia_ngang and phan_tren:
                phan_tren = False
                break
            if y < 20:
                break

    return pdf


def create_vietnamese_sheet(temp_font_path, duong_dan):

    thu_muc = tim_thu_muc(duong_dan)
    duong_dan_luu_tep = os.path.join(thu_muc, VI_filename)

    pdf = canvas.Canvas(duong_dan_luu_tep, pagesize=A4)
    pdf.line(chia_doc, 820, chia_doc, 20)  # kẻ trục

    set_font(pdf, temp_font_path, size)

    phan_trai = True
    x_pos = le_trai
    y_pos = le_tren
    for vi_char in VIETNAMESE_TEXT.split():

        chu_mau = True
        x = x_pos
        gioi_han_phai = chia_doc if phan_trai else le_phai

        while True:
            if chu_mau:
                pdf.setFillColorRGB(0, 0, 0)
                chu_mau = False
            else:
                pdf.setFillColorRGB(0.8, 0.8, 0.8)

            pdf.drawString(x, y_pos, vi_char)
            x += size + 5

            # kẻ hàng
            pdf.setDash(1, 3)
            pdf.setStrokeColorRGB(0.9, 0.9, 0.9)
            pdf.line(0, y_pos, 595, y_pos)
            pdf.setDash()

            # điều kiện dừng khi chạm trục
            if x > (gioi_han_phai - 10):
                break

        y_pos -= size + 15

        # kiểm tra điều kiện đổi trang
        if y_pos < le_duoi:
            if phan_trai:
                x_pos = chia_doc + 10
                y_pos = le_tren
                phan_trai = False
            else:
                phan_trai = True
                x_pos = le_trai
                y_pos = le_tren
                # tạo trang mới khi trang cũ đã đầy
                pdf.showPage()
                set_font(pdf, temp_font_path, size)
                pdf.line(chia_doc, 820, chia_doc, 20)

    return pdf


def is_vietnamese_font(temp_font_path):
    try:

        bang_ma = FT_Font(temp_font_path).getBestCmap()
        cac_chu_can_in = set(char for char in VIETNAMESE_TEXT if char.strip())

        for char in cac_chu_can_in:
            if ord(char) not in bang_ma:
                print(f"BỊ TRƯỢT VÌ THIẾU KÝ TỰ: '{char}' (Mã Unicode: {ord(char)})")
                return False
        return True

    # chặn lỗi
    except Exception as e:
        print(f"LỖI HỆ THỐNG KHI CHECK FONT: {e}")
        return False
