import json
from pathlib import Path
from typing import Any, Optional

import joblib


def save_model(
    model_data: Any,
    model_path: str,  # Đường dẫn model (lấy từ cfg.inference.classifier_path)
    model_info: Optional[dict] = None,
    json_path: Optional[str] = None,
) -> str:
    """
    Lưu model và JSON thông tin.
    Chỉ lưu JSON nếu người dùng chủ động cung cấp 'json_path'.
    """
    # 1. Xử lý lưu Model (Luôn thực hiện)
    m_path = Path(model_path)
    m_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_data, m_path)
    print(f"✅ Model đã lưu tại: {m_path}")

    # 2. Xử lý lưu JSON (Chỉ thực hiện khi có json_path)
    if json_path and model_info is not None:
        j_path = Path(json_path)
        # Đảm bảo thư mục chứa file JSON cũng được tạo
        j_path.parent.mkdir(parents=True, exist_ok=True)

        # Cập nhật đường dẫn tuyệt đối của model vào info để đồng bộ
        model_info["model_path"] = str(m_path.resolve())

        with open(j_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=4, ensure_ascii=False)

        print(f"✅ JSON đã lưu tại: {j_path}")
        return str(j_path)

    return str(m_path)
