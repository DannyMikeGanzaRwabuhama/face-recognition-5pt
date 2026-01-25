# Face Recognition with ArcFace ONNX and 5-Point Alignment

A CPU-friendly face recognition system using ArcFace embeddings, Haar cascade detection, and MediaPipe 5-point landmark alignment.

## Features

- ✅ **CPU-only execution** - No GPU required
- ✅ **5-point facial landmark alignment** - Robust face normalization
- ✅ **ArcFace deep learning embeddings** - State-of-the-art accuracy via ONNX
- ✅ **Open-set recognition** - Automatically rejects unknown faces
- ✅ **Multi-face real-time recognition** - Handles multiple faces simultaneously
- ✅ **Threshold tuning with FAR/FRR analysis** - Data-driven threshold selection
- ✅ **Persistent database storage** - NPZ format for embeddings
- ✅ **Re-enrollment support** - Add more samples to existing identities

## Project Structure

```
face-recognition-5pt/
├── data/
│   ├── enroll/          # Aligned face crops (generated)
│   └── db/              # Recognition database (generated)
├── models/
│   └── embedder_arcface.onnx  # ArcFace model (download required)
├── src/
│   ├── camera.py        # Camera validation
│   ├── detect.py        # Face detection test
│   ├── landmarks.py     # 5-point landmark test
│   ├── align.py         # Face alignment test
│   ├── embed.py         # Embedding extraction test
│   ├── enroll.py        # Enrollment pipeline
│   ├── evaluate.py      # Threshold evaluation
│   ├── recognize.py     # Real-time recognition
│   └── haar_5pt.py      # Core detection module
├── init_project.py      # Project structure generator
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.9 or higher
- Webcam
- Operating System: macOS, Linux, or Windows

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face-recognition-5pt.git
cd face-recognition-5pt
```

### 2. Create Project Structure

```bash
python init_project.py
```

### 3. Set Up Virtual Environment

**Create virtual environment:**
```bash
python -m venv .venv
```

**Activate:**
- **macOS/Linux:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (CMD):** `.venv\Scripts\activate.bat`

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Download ArcFace ONNX Model

⚠️ **CRITICAL STEP** - The model file is ~120MB and not included in the repository.

**Download and install:**
```bash
# Download
curl -L -o buffalo_l.zip \
"https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"

# Extract
unzip -o buffalo_l.zip

# Copy to models directory
cp w600k_r50.onnx models/embedder_arcface.onnx

# Clean up
rm -f buffalo_l.zip w600k_r50.onnx 1k3d68.onnx 2d106det.onnx det_10g.onnx genderage.onnx
```

**Validate installation:**
```bash
python -m src.embed
```

Expected output:
```
embedding dim: 512
norm(before L2): 21.85
cos(prev,this): 0.988
```

### 6. Camera Permissions

**macOS:**
1. Go to System Settings → Privacy & Security → Camera
2. Allow access for Terminal or VS Code
3. Restart terminal

**Windows/Linux:**
- Ensure no other application is using the camera

## Usage

### Step 1: Validate Your Setup

Run these tests in order to verify each component:

```bash
# Test camera access
python -m src.camera

# Test face detection
python -m src.detect

# Test 5-point landmarks
python -m src.landmarks

# Test face alignment
python -m src.align

# Test embedding extraction
python -m src.embed
```

Press `q` to quit each test.

### Step 2: Enroll Identities

```bash
python -m src.enroll
```

**Controls:**
- `SPACE` - Capture one sample
- `a` - Toggle auto-capture mode (captures every 0.25s)
- `s` - Save enrollment (requires 15+ samples)
- `r` - Reset new samples (keeps existing)
- `q` - Quit

**Tips for best results:**
- Use stable lighting
- Capture from different angles
- Include different expressions
- Move slightly left/right during capture
- Enroll at least 2 people for evaluation

### Step 3: Evaluate Threshold

```bash
python -m src.evaluate
```

This analyzes your enrolled data and suggests an optimal recognition threshold based on:
- **Genuine pairs** - Same person comparisons
- **Impostor pairs** - Different person comparisons
- **FAR** (False Accept Rate) - Target: 1%
- **FRR** (False Reject Rate) - Minimized

**Requirements:**
- At least 2 enrolled people
- At least 5 samples per person

### Step 4: Run Recognition

```bash
python -m src.recognize
```

**Controls:**
- `q` - Quit
- `r` - Reload database from disk
- `+` or `=` - Increase threshold (more accepts)
- `-` - Decrease threshold (fewer accepts)
- `d` - Toggle debug overlay

**Display shows:**
- Face bounding boxes with 5-point landmarks
- Identity labels (green = known, red = unknown)
- Distance and similarity scores
- Aligned face thumbnails (right side)
- FPS counter

## How It Works

### Pipeline Architecture

**Enrollment Pipeline:**
```
Camera → Haar Detection → MediaPipe 5pt → Alignment → ArcFace Embedding → Database
```

**Recognition Pipeline:**
```
Camera → Haar Detection → MediaPipe 5pt → Alignment → ArcFace Embedding → Matching → Label
```

### Key Components

1. **Face Detection** - Haar cascade (CPU-efficient)
2. **5-Point Landmarks** - MediaPipe FaceMesh extracts: left eye, right eye, nose, left mouth, right mouth
3. **Face Alignment** - Affine transform to 112×112 canonical pose
4. **Embedding** - ArcFace ResNet-50 produces 512-dimensional L2-normalized vectors
5. **Matching** - Cosine distance comparison (distance = 1 - similarity)
6. **Threshold** - Accept if distance ≤ threshold

### Recognition Math

- **Embedding**: 512-dimensional L2-normalized vector
- **Similarity**: `cos(a,b) = dot(a,b)` (since L2-normalized)
- **Distance**: `dist(a,b) = 1 - cos(a,b)`
- **Decision**: Accept if `dist ≤ threshold` (typically ~0.34)

## Data Storage

### Database Files

**`data/db/face_db.npz`** - Binary storage of embeddings
```python
{
  "Alice": [512-dim embedding],
  "Bob": [512-dim embedding],
  ...
}
```

**`data/db/face_db.json`** - Metadata
```json
{
  "updated_at": "2025-01-25 10:30:00",
  "embedding_dim": 512,
  "names": ["Alice", "Bob"],
  "samples_total_used": 30,
  "note": "Embeddings are L2-normalized vectors..."
}
```

### Enrollment Crops

**`data/enroll/<name>/*.jpg`** - Aligned 112×112 face crops
- Saved for inspection and evaluation
- Used for re-enrollment
- Not required at runtime

## Performance

**Typical CPU performance:**
- Enrollment: 10-15 FPS
- Recognition: 10-20 FPS (single face)
- Recognition: 8-15 FPS (2-3 faces)

**Optimizations:**
- ROI-based detection reduces computation
- Process every N frames (not every frame)
- Temporal smoothing stabilizes predictions

## Troubleshooting

### Camera not opening
- Check permissions (macOS: System Settings → Privacy → Camera)
- Try different camera index: `cv2.VideoCapture(1)` instead of `0`
- Close other apps using the camera

### Model not loading
- Verify file exists: `ls -lh models/embedder_arcface.onnx`
- Should be ~120MB
- Re-download if corrupted

### Poor recognition accuracy
- Re-enroll with more samples (20-30 per person)
- Ensure good lighting during enrollment
- Use threshold evaluation to tune threshold
- Check alignment quality: `python -m src.align`

### "FaceMesh returned none"
- Face too small - move closer to camera
- Poor lighting - improve illumination
- Face turned away - look at camera

## Re-enrollment

To add more samples to an existing identity:

```bash
python -m src.enroll
# Enter the same name
# System loads existing samples
# Capture new samples
# Press 's' to merge and save
```

## Project Background

This project is based on the book **"Face Recognition with ArcFace ONNX and 5-Point Alignment"** by Gabriel Baziramwabo (Benax Technologies Ltd · Rwanda Coding Academy).

The system emphasizes:
- Educational transparency over black-box frameworks
- CPU-first architecture for accessibility
- Modular design for debugging and extension
- Production-ready practices

## References

1. Deng et al. (2019) - ArcFace: Additive Angular Margin Loss for Deep Face Recognition
2. InsightFace Project - 2D & 3D Face Analysis
3. ONNX - Open Neural Network Exchange
4. MediaPipe - Framework for Building Perception Pipelines
5. OpenCV - Computer Vision Library

## License

This project is for educational purposes. Please respect the licenses of:
- ArcFace/InsightFace models
- MediaPipe
- OpenCV

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test thoroughly
4. Submit a pull request

## Author

**Gabriel Baziramwabo**
- Benax Technologies Ltd
- Rwanda Coding Academy

## Acknowledgments

Special thanks to students at Rwanda Coding Academy whose curiosity and feedback shaped this project's design and clarity.