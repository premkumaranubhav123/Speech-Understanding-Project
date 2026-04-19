# Robust Environmental Sound Classification (ESC)
**Using Transfer Learning with Responsible AI Analysis**

This project implements a state-of-the-art pipeline for Environmental Sound Classification (ESC) on the **UrbanSound8K** dataset. It features a dual-stream feature fusion architecture (Log-Mel Spectrograms + MFCCs) and comprehensive **Responsible AI** evaluations.

---

## 👥 Team Details
**Institution:** Indian Institute of Technology (IIT), Jodhpur  
**Course:** Speech Understanding  
**Members:**  
- Prem Kumar (B22AI031)  
- Akash Chaudhary (B22EE007)  
- V.K Santosh (B22AI049)

---

## 🚀 Features
- **SOTA Models**: Transfer learning using **ResNet-50** and **YAMNet (MobileNetV2)**.
- **Dual-Stream Processing**: Fuses 2D Spectrogram features with 1D MFCC timbral features.
- **Memory Efficient**: Sequential fold-based caching for execution on standard 16GB RAM / CPU-only systems.
- **Responsible AI Integrated**:
  - **Fairness**: Demographic Parity Index (DPI) calculation.
  - **Robustness**: Stress-testing against controlled Traffic/Crowd noise (SNR curves).
  - **Explainability**: **Grad-CAM** heatmaps visualizing model attention.

---

## 📂 Project Structure
```text
.
├── src/                    # Modular source code
│   ├── data/               # Preprocessing & Augmentation
│   ├── models/             # ResNet50 & YAMNet implementations
│   ├── responsible_ai/     # Fairness, Robustness, Grad-CAM
│   └── visualization/      # Dashboard generation
├── outputs/                # Generated artifacts
│   ├── figures/            # Plots & Grad-CAM images
│   ├── results/            # Performance metrics (JSON)
│   └── checkpoints/        # Trained model weights (.pt)
├── reports/                # Final Technical Reports (Word, LaTeX, MD)
├── presentation/           # PowerPoint Slides & Scripts
├── run_all.py              # Master orchestration script
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## 🛠️ Installation & Setup

### 1. Requirements
- Python 3.11+
- FFmpeg (for audio processing)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset Setup
Run the downloader to fetch UrbanSound8K or generate a synthetic verification set:
```bash
python src/data/download_dataset.py
```

---

## 📊 Usage

### Run Entire Pipeline
To extract features, train models, evaluate fairness, and generate the dashboard in one go:
```bash
python run_all.py
```

### Run Custom Inference
To classify an external audio file:
```bash
python src/inference.py path/to/your_audio.wav
```

---

## 🧠 Responsible AI Insights
### Fairness
The system calculates per-class F1-scores and highlights bias risks where performance significantly deviates from the mean.

### Robustness
Our models are evaluated at SNR levels from +20dB down to -5dB. The results consistently show that the dual-stream feature fusion provides superior noise resilience compared to standard approaches.

### Grad-CAM
The heatmaps found in `outputs/figures/grad_cam/` allow users to verify that the model is making decisions based on relevant acoustic textures (e.g., rhythmic pulses in drilling) rather than background silence.

---

## 📜 License
This project is developed for educational purposes at IIT Jodhpur.
 UrbanSound8K dataset remains the property of original authors (Salamon et al).
