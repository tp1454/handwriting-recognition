from src.utils.config import Config
from src.utils.handwriting_sheet_generation import (
    create_handwriting_sheet,
)

cfg = Config.from_yaml("config/default.yaml")

create_handwriting_sheet(cfg)
