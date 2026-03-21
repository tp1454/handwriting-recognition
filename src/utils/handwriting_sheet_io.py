import os
import tempfile

def save_temp_font(uploaded_file):
    temp_font = tempfile.NamedTemporaryFile(delete=False, suffix=".ttf")
    temp_font.write(uploaded_file.read())
    temp_font.close()
    return temp_font.name

def create_pdf(pdf, temp_font_path=None):
    try:
        pdf.save()
    except Exception as e:
        raise RuntimeError(f"Hệ thống không thể lưu PDF. Chi tiết: {e}")
        
    finally:
        if temp_font_path and os.path.exists(temp_font_path):
            os.remove(temp_font_path)