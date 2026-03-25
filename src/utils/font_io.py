"""
- Nhận file font từ người dùng và lưu tạm vào hệ thống
- File tạm sẽ được sử dụng trong visualize.py
- Sau khi tạo biểu đồ, xuất ra PDF
- Xóa file font tạm để giải phóng bộ nhớ
"""

import os
import tempfile


# INPUT : Nhập tệp FONT
def save_temp_font(uploaded_file):
    """
    Lưu file font (.ttf) người dùng tải lên vào vùng nhớ tạm của hệ thống.
    Trả về đường dẫn tuyệt đối của file để truyền vào ReportLab.
    """
    try:
        temp_font = tempfile.NamedTemporaryFile(delete=False, suffix=".ttf")
        temp_font.write(uploaded_file.read())
        temp_font.close()
        return temp_font.name

    except Exception as e:
        print(f"Lỗi khi lưu font tạm: {e}")
        return None


# OUTPUT : xuất tệp PDF và dọn dẹp
def create_pdf(pdf, temp_font_path=None):
    """
    Hàm này nhận biến pdf (từ create_english_sheet), thực hiện lưu file
    và dọn dẹp file font tạm thời khỏi ổ cứng.
    """
    try:
        # 1. Lưu file PDF (Nó sẽ tự lưu vào thư mục bạn đã setup ở hàm create_english_sheet)
        pdf.save()
        print("Đã xuất file PDF thành công vào thư mục đích!")

    except Exception as e:
        print(f" Lỗi khi lưu PDF: {e}")

    finally:
        # 2. Xóa file font tạm để giải phóng bộ nhớ (đáp ứng đúng yêu cầu thiết kế của bạn)
        if temp_font_path and os.path.exists(temp_font_path):
            os.remove(temp_font_path)
            print(" Đã dọn dẹp file font tạm thời.")
