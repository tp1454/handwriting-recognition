import os
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


# ==============================================================
# 1. HÀM TÌM ĐƯỜNG DẪN (GỘP CHUNG - ĐƠN GIẢN & THÔNG MINH)
# ==============================================================
def get_path(relative_path: str) -> str:
    """Tìm gốc dự án và trả về đường dẫn tuyệt đối trong 1 hàm duy nhất."""
    try:
        # Lấy vị trí file config.py hiện tại
        current = Path(__file__).resolve().parent
        root = current

        # Các dấu hiệu nhận biết thư mục gốc của dự án
        markers = (
            ".git",
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
        )

        # Leo ngược lên tìm thư mục chứa file marker
        for parent in [current] + list(current.parents):
            if any((parent / m).exists() for m in markers):
                root = parent
                break
        return str(root / relative_path)
    except NameError:
        # Dự phòng cho môi trường đặc biệt như Colab/Jupyter
        return str(Path.cwd() / relative_path)


# ==============================================================
# 2. ENUMS (BẢO VỆ KIỂU DỮ LIỆU - CHỐNG GÕ SAI)
# ==============================================================
class SchedulerType(str, Enum):
    REDUCE_ON_PLATEAU = "ReduceLROnPlateau"
    STEP_LR = "StepLR"
    COSINE = "CosineAnnealingLR"


class CheckpointMode(str, Enum):
    MIN = "min"
    MAX = "max"


# ==============================================================
# 3. CÁC KHỐI CẤU HÌNH
# ==============================================================
@dataclass
class ModelConfig:
    num_classes: int = 62
    embedding_dim: int = 128
    dropout: float = 0.3

    def __post_init__(self):
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class SchedulerConfig:
    type: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU
    factor: float = 0.5
    patience: int = 3
    min_lr: float = 0.00001


@dataclass
class SiameseConfig:
    margin: float = 1.0
    similar_ratio: float = 0.5


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 5
    save_every: int = 1
    scheduler: SchedulerConfig = field(
        default_factory=SchedulerConfig
    )
    siamese: SiameseConfig = field(default_factory=SiameseConfig)


@dataclass
class AugmentationConfig:
    enabled: bool = True
    rotation_max: float = 15
    scale_range: tuple[float, float] = (0.9, 1.1)
    translation_max: int = 3
    noise_std: float = 0.05


@dataclass
class DataConfig:
    train_path: str = "data/processed/train"
    val_path: str = "data/processed/val"
    test_path: str = "data/processed/test"
    reference_path: str = "data/reference"
    image_size: int = 28
    normalize_mean: float = 0.5
    normalize_std: float = 0.5
    augmentation: AugmentationConfig = field(
        default_factory=AugmentationConfig
    )

    def __post_init__(self):
        # Chuyển đổi thành đường dẫn tuyệt đối
        self.train_path = get_path(self.train_path)
        self.val_path = get_path(self.val_path)
        self.test_path = get_path(self.test_path)
        self.reference_path = get_path(self.reference_path)

        # 🌟 VÁ LỖI SỐ 3: Kiểm tra Data tồn tại (Fail-Fast)
        for p in [self.train_path, self.val_path, self.test_path]:
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"⚠️ Thư mục dữ liệu không tồn tại: {p}"
                )


@dataclass
class CheckpointConfig:
    dir: str = "models/checkpoints"
    save_best_only: bool = True
    monitor: str = "val_loss"
    mode: CheckpointMode = CheckpointMode.MIN

    def __post_init__(self):
        self.dir = get_path(self.dir)
        os.makedirs(self.dir, exist_ok=True)


@dataclass
class InferenceConfig:
    classifier_path: str = "models/exports/classifier.pt"
    encoder_path: str = "models/exports/encoder.pt"
    device: str = "cuda"
    threshold: float = 0.5

    def __post_init__(self):
        # Kiểm tra "tờ note" từ Sếp (Biến môi trường)
        env_model_path = os.getenv("MODEL_PATH")
        if env_model_path:
            self.classifier_path = env_model_path

        # Chốt đường dẫn tuyệt đối
        self.classifier_path = get_path(self.classifier_path)
        self.encoder_path = get_path(self.encoder_path)

        # 🌟 VÁ LỖI SỐ 2: Kiểm tra file Model có đuôi .pt tồn tại (Fail-Fast)
        # if not Path(self.classifier_path).exists():
        # raise FileNotFoundError(f"⚠️ Không tìm thấy file model: {self.classifier_path}")
        # if not Path(self.encoder_path).exists():
        #  raise FileNotFoundError(f"⚠️ Không tìm thấy file encoder: {self.encoder_path}")


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 4


@dataclass
class SheetConfig:
    save_path: str = "data/user_samples"
    font_name: str = "handwriting_sheet"
    en_filename: str = "english_sheet.pdf"
    vi_filename: str = "vietnamese_sheet.pdf"
    font_size: int = 15
    line_spacing: int = 20
    word_spacing: int = 20
    width: int = 594
    height: int = 841
    margin_left: int = 40
    margin_right: int = 555
    margin_top: int = 802
    margin_bottom: int = 40
    divide_horizontal: int = 421
    divide_vertical: int = 297
    en_text = """
    The quick brown fox jumps over the lazy dog
    THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
    """
    vi_text = """
    a à á ả ã ạ
    ă ằ ắ ẳ ẵ ặ
    â ầ ấ ẩ ẫ ậ
    e è é ẻ ẽ ẹ
    ê ề ế ể ễ ệ
    i ì í ỉ ĩ ị
    o ò ó ỏ õ ọ
    ô ồ ố ổ ỗ ộ
    ơ ờ ớ ở ỡ ợ
    u ù ú ủ ũ ụ
    ư ừ ứ ử ữ ự
    y ỳ ý ỷ ỹ ỵ

    A À Á Ả Ã Ạ
    Ă Ằ Ắ Ẳ Ẵ Ặ
    Â Ầ Ấ Ẩ Ẫ Ậ
    E È É Ẻ Ẽ Ẹ
    Ê Ề Ế Ể Ễ Ệ
    I Ì Í Ỉ Ĩ Ị
    O Ò Ó Ỏ Õ Ọ
    Ô Ồ Ố Ổ Ỗ Ộ
    Ơ Ờ Ớ Ở Ỡ Ợ
    U Ù Ú Ủ Ũ Ụ
    Ư Ừ Ứ Ử Ữ Ự
    Y Ỳ Ý Ỷ Ỹ Ỵ

    b c d đ g h k l m n p q r s t v x
    B C D Đ G H K L M N P Q R S T V X

    """

    def __post_init__(self) -> None:
        self.save_path = get_path(self.save_path)
        os.makedirs(self.save_path, exist_ok=True)


# ==============================================================
# 4. ROOT CONFIG (BẢNG ĐIỀU KHIỂN TRUNG TÂM)
# ==============================================================
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(
        default_factory=CheckpointConfig
    )
    inference: InferenceConfig = field(
        default_factory=InferenceConfig
    )
    api: APIConfig = field(default_factory=APIConfig)
    sheet: SheetConfig = field(default_factory=SheetConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return cls(
            model=_build_dataclass(
                ModelConfig, data.get("model", {})
            ),
            training=_build_dataclass(
                TrainConfig, data.get("training", {})
            ),
            data=_build_dataclass(DataConfig, data.get("data", {})),
            checkpoint=_build_dataclass(
                CheckpointConfig, data.get("checkpoint", {})
            ),
            inference=_build_dataclass(
                InferenceConfig, data.get("inference", {})
            ),
            api=_build_dataclass(APIConfig, data.get("api", {})),
            sheet=_build_dataclass(
                SheetConfig, data.get("sheet", {})
            ),
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        return cls.from_dict(load_config(config_path))


# ==============================================================
# 5. HELPERS (VÁ LỖI ENUM & LOAD CONFIG)
# ==============================================================
def _build_dataclass(dataclass_type: type, values: dict[str, Any]):
    if values is None:
        values = {}
    kwargs: dict[str, Any] = {}
    field_map = {
        field.name: field for field in fields(dataclass_type)
    }

    for name, value in values.items():
        if name not in field_map:
            continue
        field_type = field_map[name].type

        # Xử lý các Dataclass con lồng nhau
        if hasattr(field_type, "__dataclass_fields__") and isinstance(
            value, dict
        ):
            kwargs[name] = _build_dataclass(field_type, value)

        # 🌟 VÁ LỖI SỐ 1 & 4: Ép kiểu từ String (trong YAML) sang Enum (trong Code)
        elif isinstance(field_type, type) and issubclass(
            field_type, Enum
        ):
            try:
                kwargs[name] = field_type(value)
            except ValueError:
                valid = [e.value for e in field_type]
                raise ValueError(
                    f"Giá trị '{value}' không hợp lệ cho {name}. Phải là: {valid}"
                )
        else:
            kwargs[name] = value

    return dataclass_type(**kwargs)


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file YAML: {path}")
    with path.open("r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f)
    return parsed or {}


def get_log_level() -> str:
    return os.getenv("LOG_LEVEL", "INFO")
