import tkinter as tk
from tkinter import filedialog
import traceback
from src.utils.handwriting_sheet_config import (
    DUONG_DAN 
    )

from src.utils.handwriting_sheet_io import save_temp_font, create_pdf
from src.utils.handwriting_sheet_visualization import create_english_sheet, create_vietnamese_sheet, is_vietnamese_font


def main():
    print("Khởi động chương trình tạo PDF...")

    #  Khởi tạo giao diện chọn FONT
    root = tk.Tk()
    root.withdraw()

    # 1. INPUT :
    file_path = filedialog.askopenfilename(
        title="Hãy chọn một file Font chữ ",
        filetypes=[("Font TrueType", "*.ttf"), ("Tất cả các file", "*.*")],
    )

    if not file_path:
        print(" Bạn chưa chọn file font. Đang thoát chương trình...")
        return

    print(f" Đã chọn file font: {file_path}")

    try:
        with open(file_path, "rb") as uploaded_file:
            duong_dan_font_tam = save_temp_font(uploaded_file)

        if not duong_dan_font_tam:
            print("Không thể tạo file font tạm thời.")
            return

        # Kiểm tra FONT có hỗ trợ Tiếng Việt hay không
        is_vietnamese = is_vietnamese_font(duong_dan_font_tam)

        if is_vietnamese:
            print("Font có hỗ trợ Tiếng Việt")
            my_pdf = create_vietnamese_sheet(duong_dan_font_tam, DUONG_DAN)
        else:
            print("🇬🇧 Font này chỉ gõ được Tiếng Anh. Đang tạo vở Tiếng Anh...")
            my_pdf = create_english_sheet(duong_dan_font_tam, DUONG_DAN)

        # 2. OUTPUT :
        create_pdf(my_pdf, duong_dan_font_tam)
        print("QUÁ TRÌNH HOÀN TẤT!")

    except Exception as e:
        print(f" Có lỗi bất ngờ xảy ra: {e}")
        traceback.print_exc()


#
if __name__ == "__main__":
    main()
