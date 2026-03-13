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
