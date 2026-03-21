from .handwriting_sheet_config import (
    ENGLISH_TEXT, FONT_NAME, VIETNAMESE_TEXT, EN_filename, 
    VI_filename, chia_doc, chia_ngang, le_duoi, le_phai, le_trai, 
    le_tren, size
)
from pathlib import Path
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from fontTools.ttLib import TTFont as FT_Font


def tim_thu_muc(duong_dan):
    thu_muc_bat_dau = Path(__file__).resolve().parent
    cac_cap_thu_muc = [thu_muc_bat_dau] + list(thu_muc_bat_dau.parents)

    for thu_muc_cha in cac_cap_thu_muc:
        duong_dan_kiem_tra = thu_muc_cha / duong_dan

        if duong_dan_kiem_tra.is_dir():
            return str(duong_dan_kiem_tra)

    raise FileNotFoundError(
        f"Đã lùi hết cỡ nhưng không tìm thấy tổ hợp thư mục: '{duong_dan}'"
    )


def set_font(pdf, temp_font_path, font_size):
    pdfmetrics.registerFont(TTFont(FONT_NAME, temp_font_path))
    pdf.setFont(FONT_NAME, font_size)


def create_english_sheet(temp_font_path, duong_dan):
    thu_muc = tim_thu_muc(duong_dan)
    duong_dan_luu_tep = os.path.join(thu_muc, EN_filename)
    pdf = canvas.Canvas(duong_dan_luu_tep, pagesize=A4)

    set_font(pdf, temp_font_path, size)

    y = le_tren
    phan_tren = True
    for EN_text in ENGLISH_TEXT.strip().splitlines():
        mau = True
        while True:
            if stringWidth(EN_text, FONT_NAME, size) < le_phai - 40:
                x = le_trai + (
                    (le_phai - le_trai - stringWidth(EN_text, FONT_NAME, size)) / 2
                )
            else:
                x = le_trai

            if mau:
                pdf.setFillColorRGB(0, 0, 0)
                mau = False
            else:
                pdf.setFillColorRGB(0.6, 0.6, 0.6)

            


            pdf.drawString(x, y, EN_text)

            offset = size*0.1
            pdf.setStrokeColorRGB(0.9, 0.9, 0.9)
            pdf.line( 10 , y - offset, 835 , y - offset)



            y -= size + 5

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
                pdf.setFillColorRGB(0.7, 0.7, 0.7)

            pdf.drawString(x, y_pos, vi_char)
            x += size + 5

            
            

            if x > (gioi_han_phai - 10):
                break

        offset = size*0.1
        pdf.setStrokeColorRGB(0.9, 0.9, 0.9)
        pdf.line( x_pos-5 , y_pos - offset, x - 5  , y_pos - offset)

        pdf.setStrokeColorRGB(0.2, 0.2, 0.2)
        pdf.line(chia_doc, 830 , chia_doc, 10)

        y_pos -= size + 15

        if y_pos < le_duoi:
            if phan_trai:
                x_pos = chia_doc + 10
                y_pos = le_tren
                phan_trai = False
            else:

                phan_trai = True
                x_pos = le_trai
                y_pos = le_tren
                pdf.showPage()
                set_font(pdf, temp_font_path, size)
                

    return pdf


def is_vietnamese_font(temp_font_path):
    try:
        bang_ma = FT_Font(temp_font_path).getBestCmap()
        cac_chu_can_in = set(char for char in VIETNAMESE_TEXT if char.strip())

        for char in cac_chu_can_in:
            if ord(char) not in bang_ma:
                return False
        return True

    except Exception as e:
        raise RuntimeError(f"Lỗi hệ thống khi đọc dữ liệu font: {e}")