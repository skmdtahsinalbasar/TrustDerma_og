# TrustDerma Web App

> AI-assisted skin lesion analysis with confidence scoring, uncertainty estimation, trust thresholds, and Grad-CAM++ explainability.

##  Features
-  **EfficientNetV2-S** backbone trained on 7 dermatology classes
-  384x384 image preprocessing pipeline
-  **Trust Score** (locked to model confidence) with 0.80 threshold
-  Predictive entropy (uncertainty) measurement
-  Top-3 predictions with probability breakdown
-  **Grad-CAM++** visual explainability heatmaps
-  Image quality/sharpness scoring
-  Clinical-style downloadable report (TXT)

##  Classes
| Code | Full Name |
|------|-----------|
| `akiec` | Actinic Keratoses / Intraepithelial Carcinoma |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic Nevus |
| `vasc` | Vascular Lesion |

---

##  Quick Start (Local Run)

### 1. Clone the Repository
```bash
git clone https://github.com/skmdtahsinalbasar/TrustDerma.git
cd TrustDerma
```

### 2. Add Your Trained Model
Place your `best_baseline.pth` file inside the `models/` folder:
```
TrustDerma/
 models/
     best_baseline.pth    <- PUT YOUR MODEL HERE
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Python Version:** 3.10+ recommended (tested on 3.13)

### 4. Run the App
```bash
streamlit run app.py
```

The app will open automatically at: **http://localhost:8501**

---

##  Project Structure
```
TrustDerma/
├── app.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Ignore patterns
├── .streamlit/
│   └── config.toml             # Streamlit configuration
└── models/
    ├── .gitkeep                # Keep folder in repo
    ├── README.md               # Model instructions
    └── best_baseline.pth       # YOUR MODEL (not in repo - add locally)
```

---

##  Configuration

### Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `TRUSTDERMA_MODEL` | `models/best_baseline.pth` | Path to your trained model checkpoint |

### Model Format
The checkpoint can be saved as either:
- **Raw state dict:** `torch.save(model.state_dict(), "best_baseline.pth")`
- **Full checkpoint:** `torch.save({"model_state_dict": model.state_dict(), ...}, "best_baseline.pth")`

---

##  Troubleshooting

### PermissionError: `.streamlit` directory
Create the directory manually:
```bash
mkdir -p ~/.streamlit
mkdir -p .streamlit
```
Or disable usage stats in `.streamlit/config.toml`:
```toml
[browser]
gatherUsageStats = false
```

### ValueError: operands could not be broadcast together (Grad-CAM shapes)
This is fixed in the current version of `app.py`. If you see it, ensure `cv2.resize()` is being called on the heatmap before overlaying.

### Model not loading
- Verify the file exists at `models/best_baseline.pth`
- Use absolute path or override with `TRUSTDERMA_MODEL=/full/path/best_baseline.pth`
- Ensure the architecture is `tf_efficientnetv2_s` with 7 output classes

---

##  Production Deployment

### Deploy to Streamlit Community Cloud
1. Push this repo to GitHub (already done!)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repo
4. Set **Main file path:** `app.py`
5. **Important:** Add your `best_baseline.pth` via Secrets or mount a storage bucket (the model file is NOT in this repo due to GitHub file size limits)
6. Click **Deploy**

### Alternative: Hugging Face Spaces / Render / Railway
Same pattern  containerize with Docker or use Python buildpacks, and ensure the model file is accessible at runtime.

---

##  Medical Disclaimer

**This is a research/decision-support prototype. It does NOT establish a definitive medical diagnosis and must NOT replace professional medical assessment, dermoscopy, biopsy, or clinical judgment.**

MC Dropout is not used as the primary Trust Score because the original experiment showed essentially zero stochastic variation.
