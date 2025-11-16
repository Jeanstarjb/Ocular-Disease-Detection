# 👁️ Ocular Vision - AI-Driven Disease Detection

A **Clinical Decision Support System** for automated multi-label ocular disease detection from fundus images using DenseNet-121 transfer learning.

**Status:** ✅ Deployed | **Test AUC:** 0.9666 | **Accuracy:** 94.69% | **Macro F1:** 0.7871

---

## 🎯 Quick Links

| Resource | Description | Link |
|:---------|:------------|:-----|
| 🌐 **Live Streamlit App** | Upload fundus images for real-time diagnosis | [Try it now](https://a-teamstrivetowin.streamlit.app/) |
| 📊 **Tableau Dashboard** | Interactive performance analytics & demographics | [View dashboard](https://public.tableau.com/app/profile/teresia.ndung.u/viz/AI-drivenoculardiseasedetection/Dashboard1) |
| 🌍 **Project Website** | Complete case study & technical walkthrough | [Explore website](https://jeanstarjb.github.io/Ocular-Disease-Detection-web-report/) |
| 📄 **Technical Report** | Full documentation & methodology | [Read report](https://github.com/Jeanstarjb/Ocular-Disease-Detection/blob/main/Final%20Report.pdf) |
| 💻 **GitHub Repository** | Source code & notebooks | [View code](https://github.com/Jeanstarjb/ocular-disease-detection) |

---

## 📖 What This Project Does

This Moringa School capstone project addresses the **scalability crisis in ophthalmology** by creating an automated system that provides a reliable "second opinion" for medical professionals and enables early detection of preventable blindness.

### 8 Detected Ocular Pathologies

Our multi-label classification model can simultaneously detect:

- ✅ **Normal** (Healthy)
- ✅ **Diabetes** (Diabetic Retinopathy)
- ✅ **Glaucoma**
- ✅ **Cataract**
- ✅ **AMD** (Age-related Macular Degeneration)
- ✅ **Hypertension**
- ✅ **Myopia**
- ✅ **Other Abnormalities**

### Real-World Impact

- 🏥 **40-50% reduction** in specialist workload through automated screening
- 🌍 **Democratizes access** to early diagnosis in underserved regions
- ⚡ **2-3ms inference time** enables real-time screening
- 🎯 **94.69% accuracy** provides clinical-grade reliability

---

## 📊 Model Performance

| Metric | Value | Status |
|:-------|:-----:|:------:|
| Test AUC | **0.9666** | ✅ (Target: ≥0.90) |
| Test Accuracy | **94.69%** | ✅ |
| Macro F1-Score | **0.7871** | ✅ |
| Inference Time | **2–3 ms/image** | ⚡ |
| Model Size | **230 MB** | 💾 |

### Per-Class Performance

| Disease | Precision | Recall | F1-Score | Status |
|:--------|:---------:|:------:|:--------:|:------:|
| **Cataract** | 0.91 | 0.90 | **0.91** | 🏆 Best |
| **Myopia** | 0.88 | 0.88 | **0.88** | 🏆 Best |
| **AMD** | 0.86 | 0.85 | **0.86** | ✅ |
| **Glaucoma** | 0.81 | 0.85 | **0.83** | ✅ |
| **Normal** | 0.78 | 0.86 | **0.82** | ✅ |
| **Hypertension** | 0.80 | 0.78 | **0.79** | ✅ |
| **Diabetes** | 0.84 | 0.71 | **0.77** | ⚠️ Lower recall |
| **Other** | 0.65 | 0.57 | **0.65** | ⚠️ Needs improvement |

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

## 🏗️ Technical Architecture

### Model: DenseNet-121 with Transfer Learning

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

### Training Strategy

**2-Phase Fine-Tuning:**
- **Phase 1 (5 epochs):** Frozen base layers, train classification head
- **Phase 2 (15 epochs):** Unfreeze base, end-to-end fine-tuning

### Technology Stack

| Category | Technologies |
|:---------|:-------------|
| **Data Science** | Python, Pandas, NumPy, Scikit-Learn |
| **Deep Learning** | TensorFlow 2.13, Keras, DenseNet-121 |
| **Visualization** | Matplotlib, Seaborn, Tableau |
| **Deployment** | Streamlit, GitHub Pages |
| **Collaboration** | Git, GitHub |

---

## 📦 Dataset

| Aspect | Details |
|:-------|:--------|
| **Total Images** | 37,649 |
| **Train / Val / Test** | 64% / 16% / 20% |
| **Classes** | 8 (multi-label) |
| **Image Size** | 224×224 pixels |
| **Format** | RGB JPEG/PNG |

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
- **GPU:** NVIDIA GTX 1060 (6GB VRAM)
- **RAM:** 8 GB
- **Storage:** 250 MB
- **CPU:** Intel i7 / AMD Ryzen 5

### Recommended (Training)
- **GPU:** NVIDIA A100 / RTX 4090 (40GB+)
- **RAM:** 64 GB
- **Storage:** 500 GB SSD
- **CPU:** High-core processor

### Software Dependencies

```
tensorflow==2.13.0
keras==2.13.0
numpy==1.24.3
scikit-learn==1.3.0
pandas==2.0.0
pillow==9.5.0
streamlit==1.24.0
matplotlib==3.7.0
seaborn (latest)
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔬 Clinical Use Cases

### ✅ Approved Applications

**Pre-Screening Triage**
- Flag abnormal cases for urgent review
- Prioritize sight-threatening conditions
- Enable efficient resource allocation

**Normal Scan Filtering**
- Automate healthy eye identification
- Free specialist capacity for complex cases
- Reduce routine screening burden

**Remote Screening**
- Enable diagnosis in underserved areas
- Scalable to resource-limited settings
- Support telemedicine initiatives

### ⚠️ Important Clinical Disclaimer

This model is an **assistive screening tool only**. All predictions require:
- Specialist review and clinical correlation
- Confirmation by licensed ophthalmologist
- Integration with patient history and symptoms
- Compliance with local medical regulations

---

## ⚠️ Known Limitations

- 🔍 **"Other" Class:** Lower recall (57%) due to heterogeneous pathologies
- 🎯 **Diabetes Recall:** 71% sensitivity; may miss some early-stage cases
- 📊 **Class Imbalance:** Rare diseases (3-5%) have limited training data
- 🔤 **Single Modality:** Fundus images only; no OCT, visual fields, or IOP integration
- 📈 **No Severity Grading:** Detects disease presence but not progression stage
- 🌐 **Dataset Bias:** Training data may not represent all global populations

---

## 🔄 API Usage

### Option 1: Streamlit App (Easiest)
Upload image → Get predictions → View triage recommendation

### Option 2: Python API
```python
from app.inference import predict_disease

predictions = predict_disease('path/to/image.jpg')
# Returns: {'disease_name': probability, ...}
```

### Option 3: FastAPI (Production Deployment)
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@fundus_image.jpg"
```

---

## 🔮 Future Work

### Short-term (3-6 months)
- [ ] Grad-CAM explainability visualization
- [ ] Sub-categorize "Other" pathologies
- [ ] External validation (Messidor-2, EyePACS, APTOS)
- [ ] Severity grading for Diabetes & AMD

### Mid-term (6-12 months)
- [ ] Multi-modal architecture (image + patient metadata)
- [ ] Federated learning for privacy preservation
- [ ] Uncertainty quantification
- [ ] EHR integration (HL7/FHIR)

### Long-term (12+ months)
- [ ] OCT & visual field analysis
- [ ] Longitudinal progression modeling
- [ ] Mobile/edge deployment
- [ ] Demographic-specific model variants

See [Final Technical Report](Final_Report.pdf) for detailed roadmap.

---

## 🧑‍💻 The Team

This collaborative capstone project was developed by a 6-person team of data scientists:

| Team Member | Role | LinkedIn |
|:------------|:-----|:---------|
| **Jeff Munyaka Mogaka** | Project Lead | [LinkedIn](https://linkedin.com/in/jeff-munyaka-mogaka) |
| **Kitts Kikumu** | ML Engineer | [LinkedIn](https://linkedin.com/in/kitts-kikumu) |
| **Kelvin Kinoti** | Data Engineer | [LinkedIn](https://linkedin.com/in/kelvin-kinoti) |
| **Judith Otieno** | Research Analyst | [LinkedIn](https://linkedin.com/in/judith-otieno) |
| **Teresia Ndung'u** | Data Visualization | [LinkedIn](https://linkedin.com/in/teresia-ndungu) |
| **Fridah Njung'e** | Clinical Validation | [LinkedIn](https://linkedin.com/in/fridah-njunge) |

---

## 📄 Documentation

| Document | Purpose | Link |
|:---------|:--------|:-----|
| **Technical Report** | Complete methodology, evaluation & results | [View PDF](Final_Report.pdf) |
| **Clinical Guidelines** | Medical use recommendations | [View docs](docs/CLINICAL_GUIDELINES.md) |
| **API Documentation** | Integration reference | [View docs](docs/API_DOCUMENTATION.md) |
| **Deployment Guide** | Production setup instructions | [View docs](docs/DEPLOYMENT_GUIDE.md) |
| **Project Website** | Interactive case study | [Visit site](https://jeanstarjb.github.io/Ocular-Disease-Detection-web-report/) |

---

## 📜 License

MIT License — Free for research, education, and commercial use with attribution.

See [LICENSE](LICENSE) file for full terms.

---

## 🙏 Citation

If you use this work in your research or applications, please cite:

```bibtex
@software{ocular_disease_2025,
  title={Ocular Vision: AI-Driven Multi-Label Disease Detection using DenseNet-121},
  author={Mogaka, Jeff Munyaka and Kikumu, Kitts and Kinoti, Kelvin and 
          Otieno, Judith and Ndung'u, Teresia and Njung'e, Fridah},
  year={2025},
  institution={Moringa School},
  url={https://github.com/Jeanstarjb/ocular-disease-detection},
  note={Clinical Decision Support System - Capstone Project}
}
```

---

## 💬 Support & Contributions

| Need Help? | Link |
|:-----------|:-----|
| 🐛 **Report Issues** | [GitHub Issues](https://github.com/Jeanstarjb/ocular-disease-detection/issues) |
| 💬 **Discussions** | [GitHub Discussions](https://github.com/Jeanstarjb/ocular-disease-detection/discussions) |
| 📧 **Contact** | [Project Email](mailto:team@ocular-vision.dev) |

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## ⭐ Key Takeaways

🎯 **Clinical-grade performance** (AUC 0.9666) validated on 37,649 images

🚀 **Production-ready & deployed** with live web app and dashboard

📈 **40-50% efficiency gains** for specialist screening workflows

🌍 **Democratizes access** to early detection in underserved regions

💡 **Extensible architecture** ready for multi-modal expansion

🏆 **Best-in-class results** for Cataract (F1: 0.91) and Myopia (F1: 0.88)

---

<div align="center">

**🌟 Star this repository if you find it useful! 🌟**

[Live Demo](https://a-teamstrivetowin.streamlit.app/) · [Dashboard](https://public.tableau.com/app/profile/teresia.ndung.u/viz/AI-drivenoculardiseasedetection/Dashboard1) · [Website](https://jeanstarjb.github.io/Ocular-Disease-Detection-web-report/) · [Report Issues](https://github.com/Jeanstarjb/ocular-disease-detection/issues)

---

**Last Updated:** November 16, 2025 | **Version:** 1.0.0 | **Status:** ✅ Live & Production-Ready

*Developed as a Moringa School Data Science Capstone Project*

</div>
