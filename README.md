# Handwriting Recognition & Similarity Scoring

A machine learning system that classifies handwritten characters (A-Z, a-z, 0-9) and scores their similarity against reference fonts.

## 🎯 Project Goals

1. **Character Classification** — Predict which character was written
2. **Similarity Scoring** — Compare handwriting style to a reference

## 🏗️ Architecture

```
Handwriting Input (image)
        │
        ▼
Preprocessing (grayscale, resize, normalize)
        │
        ├──► Character Classifier (CNN) → "A", "B", ...
        │
        └──► Feature Encoder (Siamese) → similarity score
```

## 📁 Project Structure

```
handwriting-recognition/
├── config/                 # Configuration files
├── data/
│   ├── raw/               # Original datasets (EMNIST, etc.)
│   ├── processed/         # Preprocessed images
│   ├── reference/         # Reference font samples
│   └── user_samples/      # User-submitted handwriting
├── models/
│   ├── checkpoints/       # Saved model weights
│   └── exports/           # Production-ready models
├── notebooks/             # Jupyter notebooks for exploration
├── src/
│   ├── data/              # Data loading & preprocessing
│   ├── models/            # Model architectures
│   ├── training/          # Training loops & losses
│   ├── inference/         # Prediction & similarity scoring
│   └── utils/             # Helper functions
├── tests/                 # Unit tests
├── api/                   # FastAPI backend (deployment)
└── web/                   # Web frontend
```

## 🔧 Tech Stack

- **Framework**: PyTorch / TensorFlow
- **Preprocessing**: OpenCV, Pillow
- **API**: FastAPI
- **Frontend**: ...
- **Deployment**: Docker, AWS/GCP
## 📊 Datasets


## 🚀 Training Pipeline

1. **Phase 1**: Train CNN classifier on EMNIST
2. **Phase 2**: Extract embeddings (remove softmax)
3. **Phase 3**: Train Siamese network for similarity

## 🖥️ Run API + Web (Scoring + Sheet Creation)

### 1. Start FastAPI backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start separate web app

```bash
cd web
python3 -m http.server 5173
```

### 3. Open browser

- http://localhost:5173

### Main endpoints for web workflows

- `POST /sheet/score`
- Form field `file`: image upload (png/jpg/webp)

- `GET /sheet/options`
- Returns create-form defaults, language options, and available server fonts

- `POST /sheet/create`
- Supports either `server_font` or uploaded `font_file` plus optional `custom_text` and advanced layout fields

- `GET /sheet/files/{filename}`
- Serves generated PDF/PNG artifacts for preview and download

Response includes:
- `overall_score` (0-100)
- `detected_rows`: number of row groups detected from EasyOCR boxes
- `extracted_boxes[]`: middle-step EasyOCR extraction output (`index`, `box`, `label`, `confidence`)
- `rows[]` with mandatory `ocr_label`, plus `source_box`, `split_count`, `was_split`, `row_score`, and `segment_scores` (comparison-only scores vs split index 0)

Notes:
- `/sheet/score` requires EasyOCR initialization; if unavailable, the API returns `503 Service Unavailable`.
- Web app always shows row `ocr_label` with row score; extracted box overlay can be toggled on/off in preview.
- Server-hosted create fonts are discovered from `sheet.fonts_dir` in `config/default.yaml`.

## 🐳 Run With Docker (Dev)

### 1. Build images

```bash
docker compose build
```

### 2. Start services

```bash
docker compose up -d
```

or with Makefile:

```bash
make docker-up
```

### 3. Open services

- API docs: http://localhost:8000/docs
- API health: http://localhost:8000/health
- Web app: http://localhost:5173

### 4. Stop services

```bash
docker compose down
```

or with Makefile:

```bash
make docker-down
```

## 📝 TODO

- [ ] Set up environment
- [ ] Download and preprocess EMNIST
- [ ] Implement CNN classifier
- [ ] Train classification model
- [ ] Implement Siamese network
- [ ] Train similarity model
- [ ] Build inference pipeline
- [ ] Create web interface
- [ ] Deploy API

## 📄 License

MIT
