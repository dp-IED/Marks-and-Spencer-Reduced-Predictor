# M&S Reduced Items Prediction System

An automated system that processes M&S store videos to detect yellow reduced stickers, identifies products using YOLO + CLIP + VLLM with catalog RAG, stores detections in a dataset, and trains a probability prediction model for which products will be reduced.

## 🎯 Overview

This system helps predict which M&S products are likely to be reduced by:

1. **Video Processing**: Extracts frames and detects yellow reduced stickers using YOLO
2. **Product Identification**: Matches detected items to catalog using CLIP + VLLM with RAG
3. **Data Collection**: Stores detection patterns in a database
4. **Prediction Model**: Trains on historical data to predict reduction probabilities

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  Frame Extractor │───▶│ Sticker Detector│───▶│ Product Matcher │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
                                                                              │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐         ▼
│   Predictions   │◀───│ Prediction Model │◀───│  Data Collector │◀───┌─────────────────┐
└─────────────────┘    └──────────────────┘    └─────────────────┘    │   Database      │
                                                                      └─────────────────┘
```

## 🚀 Quick Start


### 1. Setup

```bash
# Precompute catalog embeddings (one-time setup)
python scripts/precompute_embeddings.py

# Start API server
python src/api/main.py
```

### 2. Process Videos

```bash
# Process a single video
python scripts/process_video.py --video path/to/video.mp4 --branch "London Camden"

# Train prediction model (after collecting data)
python scripts/train_prediction_model.py --min-samples 500
```

## 📁 Project Structure

```
MetS/
├── catalog/                          # M&S catalog CSV files
├── data/
│   ├── raw_videos/                   # Uploaded videos
│   ├── training_images/              # Photos for YOLO training
│   ├── annotations/                  # YOLO annotations
│   └── detections.db                 # SQLite database
├── models/
│   ├── yolo_sticker_detector/        # Fine-tuned YOLO model
│   ├── product_embeddings/           # CLIP embeddings cache
│   └── reduction_predictor/          # Prediction model
├── src/
│   ├── video_processing/             # Frame extraction & YOLO
│   ├── product_identification/       # CLIP + RAG + VLLM
│   ├── dataset/                      # Database operations
│   ├── training/                     # YOLO training
│   ├── prediction/                   # ML model training
│   └── api/                          # FastAPI server
├── scripts/                          # CLI tools
├── requirements.txt
├── config.yaml
└── README.md
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
video_processing:
  frame_extraction_rate: 1.0 # frames per second
  blur_threshold: 100.0
  max_frame_size: 1280

yolo:
  model_path: models/yolo_sticker_detector/best.pt
  confidence_threshold: 0.5
  iou_threshold: 0.45

clip:
  model_name: ViT-L/14
  embedding_cache: models/product_embeddings/
  similarity_threshold: 0.65

vllm:
  model_name: llava-v1.6-vicuna-7b
  temperature: 0.2
  max_tokens: 100

prediction:
  model_path: models/reduction_predictor/model.pkl
  retrain_frequency_days: 7
  min_training_samples: 500
```

## 📊 API Endpoints

### Video Processing

- `POST /api/v1/videos/upload` - Upload video
- `GET /api/v1/videos/{id}/status` - Check processing status
- `POST /api/v1/videos/{id}/process` - Process video

### Data Access

- `GET /api/v1/detections` - Query historical detections
- `GET /api/v1/predictions` - Get reduction predictions
- `GET /api/v1/catalog/products` - Search catalog

### System

- `GET /api/v1/health` - Health check
- `GET /api/v1/stats` - System statistics
- `POST /api/v1/predictions/refresh` - Retrain model

## 🛠️ Development

### Testing

```bash
# Test video processing
python scripts/process_video.py --video test_video.mp4 --branch "Test Branch"

# Test API
curl http://localhost:8000/api/v1/health

# Test model training
python scripts/train_prediction_model.py --start-date 2024-01-01 --end-date 2024-12-31
```

## 📈 Performance

### Target Performance

- **YOLO Detection**: mAP > 0.85 for yellow stickers
- **Product Identification**: > 90% accuracy on manual validation
- **Prediction Model**: AUC-ROC > 0.75 for reduction probability
- **Processing Speed**: 10-min video in < 5 minutes

### Data Quality

- Record during peak hours (afternoon/evening)
- Include various product categories
- Ensure good lighting and stable camera
- Avoid blurry or obstructed views

## 📚 Technical Details

### Models Used

- **YOLOv8**: Yellow sticker detection
- **CLIP (ViT-L/14)**: Product similarity
- **VLLM (LLaVA)**: Final product matching
- **XGBoost**: Reduction probability prediction

### Database Schema

- `detections`: Individual detection records
- `videos`: Video metadata and processing status
- `model_metrics`: Model performance tracking

### Data Flow

1. Video → Frame extraction → YOLO detection
2. Extracted frames → CLIP embedding → RAG search
3. Top candidates → VLLM confirmation → Product ID
4. Detection → Database storage → Feature engineering
5. Historical data → Model training → Predictions

## 📄 License

This project is for educational and research purposes. Please respect M&S's terms of service and privacy policies.

## 🙏 Acknowledgments

- OpenAI CLIP for visual similarity
- Ultralytics YOLO for object detection
- ChromaDB for vector search
- FastAPI for API framework
