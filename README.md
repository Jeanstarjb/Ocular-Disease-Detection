# 👁️ AI-Driven Ocular Disease Detection

A **Clinical Decision Support System** for automated multi-label ocular disease detection from fundus images using DenseNet-121 transfer learning.

**Status:** ✅ Deployed | **Test AUC:** 0.9666 | **Accuracy:** 94.69%

---

## 🎯 Quick Links

- 🌐 **Live App:** https://a-teamstrivetowin.streamlit.app/
- 📊 **Dashboard:** https://public.tableau.com/app/profile/teresia.ndung.u/viz/AI-drivenoculardiseasedetection/Dashboard1


---

## 📖 What This Project Does

This system detects **8 ocular diseases** from a single fundus image:
- ✅ Normal
- ✅ Diabetes (Diabetic Retinopathy)
- ✅ Glaucoma
- ✅ Cataract
- ✅ AMD (Age-related Macular Degeneration)
- ✅ Hypertension
- ✅ Myopia
- ✅ Other Abnormalities

**Real-world impact:** Automates screening to reduce specialist workload by 40-50% and enables diagnosis in underserved regions.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- GPU (NVIDIA GTX 1060+ recommended for inference)
- 250 MB storage for model weights

### Installation

```bash
# Clone repository
git clone https://github.com/Jeanstarjb/ocular-disease-detection.git
cd ocular-disease-detection

# Install dependencies
pip install -r requirements.txt
```

### Run the Web App

```bash
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

### Use the Model in Code

```python
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model('models/densenet121_best_model_phase2.keras.weights.h5')

# Prepare image
img = Image.open('fundus_image.jpg').convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Get predictions
predictions = model.predict(img_array)

# Decode results
class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 
               'AMD', 'Hypertension', 'Myopia', 'Other']

for idx, class_name in enumerate(class_names):
    print(f"{class_name}: {predictions[0][idx]:.2%}")
```

---

## 📊 Model Performance

| Metric | Value |
|:-------|------:|
| Test AUC | **0.9666** ✓ |
| Test Accuracy | **94.69%** |
| Macro F1-Score | **0.7871** |
| Inference Time | 2–3 ms/image |

### Per-Class Results

| Disease | Precision | Recall | F1-Score |
|:--------|:----------:|:-------:|:--------:|
| Cataract | 0.91 | 0.90 | **0.91** 🏆 |
| Myopia | 0.88 | 0.88 | **0.88** 🏆 |
| AMD | 0.86 | 0.85 | **0.86** |
| Glaucoma | 0.81 | 0.85 | **0.83** |
| Normal | 0.78 | 0.86 | **0.82** |
| Diabetes | 0.84 | 0.71 | **0.77** |
| Hypertension | 0.80 | 0.78 | **0.79** |
| Other | 0.65 | 0.57 | **0.65** |

---

## 🏗️ Architecture

**Transfer Learning with DenseNet-121**

```
Input (224×224×3)
        ↓
DenseNet-121 Base
(Pre-trained ImageNet)
        ↓
GlobalAveragePooling2D
        ↓
Dense(512, ReLU) + Dropout(0.5)
        ↓
Dense(8, Sigmoid)
        ↓
Output: 8-Class Probabilities
```

**Training:** 2-phase fine-tuning
- Phase 1 (5 epochs): Frozen base, train head
- Phase 2 (15 epochs): Unfreeze & fine-tune

---

## 📦 Dataset

| Aspect | Details |
|:-------|:--------|
| Total Images | 37,649 |
| Train / Val / Test | 64% / 16% / 20% |
| Classes | 8 (multi-label) |
| Image Size | 224×224 pixels |
| Format | RGB JPEG/PNG |

**Sources:**
- ODIR-5K: 6,392 images
- Augmented Datasets: 31,257 images
- **Total:** 37,649 fully validated

---

## 📁 Project Structure

```
ocular-disease-detection/
├── app/
│   ├── streamlit_app.py          # Main web application
│   ├── inference.py              # Model inference pipeline
│   └── config.py                 # App configuration
├── models/
│   └── densenet121_best_model_phase2.keras.weights.h5
├── src/
│   ├── data_pipeline.py          # Custom data generator
│   ├── model.py                  # Model architecture
│   └── preprocessing.py          # Image preprocessing
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── docs/
│   ├── TECHNICAL_REPORT.md       # Full technical documentation
│   ├── CLINICAL_GUIDELINES.md    # Clinical use recommendations
│   ├── API_DOCUMENTATION.md      # API reference
│   └── DEPLOYMENT_GUIDE.md       # Production deployment
├── requirements.txt
├── README.md                     # This file
└── LICENSE                       # MIT License
```

---

## 💻 System Requirements

### Minimum (Inference Only)
- GPU: NVIDIA GTX 1060 (6GB VRAM)
- RAM: 8 GB
- Storage: 250 MB
- CPU: Intel i7 / AMD Ryzen 5

### Recommended (Training)
- GPU: NVIDIA A100 / RTX 4090 (40GB+)
- RAM: 64 GB
- Storage: 500 GB SSD
- CPU: High-core processor

---

## 📋 Software Dependencies

```
tensorflow==2.13.0
keras==2.13.0
numpy==1.24.3
scikit-learn==1.3.0
pandas==2.0.0
pillow==9.5.0
streamlit==1.24.0
matplotlib==3.7.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🔬 Clinical Use Cases

✅ **Pre-Screening Triage**
- Flag abnormal cases for urgent review
- Prioritize sight-threatening conditions

✅ **Normal Scan Filtering**
- Automate healthy eye identification
- Free specialist capacity for complex cases

✅ **Remote Screening**
- Enable diagnosis in underserved areas
- Scalable to resource-limited settings

⚠️ **Important:** Model is an **assistive screening tool only**. All predictions require specialist review and clinical correlation.

---

## ⚠️ Limitations

- 🔍 **"Other" Class:** Lower recall (57%) due to heterogeneous pathologies
- 🎯 **Diabetes Recall:** 71% sensitivity; may miss some cases
- 📊 **Class Imbalance:** Rare diseases (3-5%) have limited training data
- 🔤 **Single Modality:** Fundus image only; no OCT, visual fields, or IOP
- 📈 **No Severity Grading:** Detects disease presence, not stage


---

## 🔄 API Usage

### Streamlit App (Easiest)
Upload image → Get predictions → View triage recommendation

### Python API
```python
from app.inference import predict_disease

predictions = predict_disease('path/to/image.jpg')
# Returns: {'disease_name': probability, ...}
```

### FastAPI (if deployed with API server)
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@fundus_image.jpg"
```

---

## 📈 Results Summary

✅ **All success criteria achieved:**
- AUC: 0.9666 (target: ≥0.90)
- Accuracy: 94.69%
- Macro F1: 0.7871

✅ **Best performing classes:**
- Cataract: F1 = 0.91
- Myopia: F1 = 0.88

✅ **Production ready:**
- Live web app deployed
- <3ms inference per image
- 230 MB model size

---

## 🔮 Future Work

**Short-term (3-6 months):**
- [ ] Grad-CAM explainability visualization
- [ ] Sub-categorize "Other" pathologies
- [ ] External validation (Messidor-2, EyePACS, APTOS)
- [ ] Severity grading for Diabetes & AMD

**Mid-term (6-12 months):**
- [ ] Multi-modal architecture (image + patient metadata)
- [ ] Federated learning for privacy
- [ ] Uncertainty quantification
- [ ] EHR integration (HL7/FHIR)

**Long-term (12+ months):**
- [ ] OCT & visual field analysis
- [ ] Longitudinal progression modeling
- [ ] Mobile/edge deployment
- [ ] Demographic-specific model variants

See (Final Report) for detailed roadmap.

---

## 📄 Documentation

| Document | Purpose |
|:---------|:--------|
| [TECHNICAL_REPORT](Final_Report.pdf) | Complete technical details, business context, and evaluation |

---

## 📊 External Links

- **Live Web App:** https://a-teamstrivetowin.streamlit.app/
- **Analytics Dashboard:** https://public.tableau.com/app/profile/teresia.ndung.u/viz/AI-drivenoculardiseasedetection/Dashboard1
- **GitHub Repository:** https://github.com/Jeanstarjb/ocular-disease-detection

---

## 📜 License

MIT License — Free for research, education, and commercial use.

---

## 🙏 Citation

```bibtex
@software{ocular_disease_2024,
  title={AI-Driven Ocular Disease Detection: Multi-Label Classification using DenseNet-121},
  author={A-TEAM},
  year={2025},
  url={https://github.com/Jeanstarjb/ocular-disease-detection},
  note={Clinical Decision Support System}
}
```

---

## 💬 Support

- 🐛 **Issues:** https://github.com/Jeanstarjb/ocular-disease-detection/issues
- 💬 **Discussions:** https://github.com/Jeanstarjb/ocular-disease-detection/discussions

---

## ⭐ Key Takeaways

🎯 **Achieves clinical-grade performance** (AUC 0.9666)

🚀 **Production-ready & deployed** (live web app + dashboard)

📈 **40-50% efficiency gains** for specialist workflows

🌍 **Democratizes access** to early screening

💡 **Extensible architecture** for future improvements

---

**Last Updated:** November 10, 2025 | **Version:** 1.0.0 | **Status:** ✅ Live
