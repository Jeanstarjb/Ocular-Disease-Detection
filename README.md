# 👁️ Ocular Disease Detection

A multi-label deep learning classifier that detects 8 ocular pathologies from retinal fundus images, built with DenseNet-121 transfer learning.

**Test AUC:** 0.9666 | **Test Accuracy:** 94.69% | **Macro F1:** 0.82

Moringa School Data Science capstone project, developed by a 6-person team. I served as project lead.

---

## Quick Links

| Resource | Description | Link |
|:---------|:------------|:-----|
| 🌐 Live App | Upload a fundus image for a real-time prediction | [a-teamstrivetowin.streamlit.app](https://a-teamstrivetowin.streamlit.app/) |
| 📊 Tableau Dashboard | Interactive performance & demographic breakdown | [View dashboard](https://public.tableau.com/app/profile/teresia.ndung.u/viz/AI-drivenoculardiseasedetection/Dashboard1) |
| 🌍 Project Website | Full case study and write-up | [View site](https://jeanstarjb.github.io/Ocular-Disease-Detection-web-report/) |
| 📄 Final Report | Complete methodology and results (PDF) | [Final Report.pdf](Final%20Report.pdf) |
| 🖥️ Presentation | Project slide deck (PDF) | [Oculus_Presentation.pdf](Oculus_Presentation.pdf) |

---

## What This Project Does

Manual screening for ocular disease doesn't scale to the number of people who need it, especially in regions with few ophthalmologists. This project trains a multi-label CNN that flags disease indicators directly from a fundus (retinal) photo, intended as a triage / second-opinion tool rather than a diagnostic replacement.

The model predicts across 8 classes simultaneously (an eye can show more than one condition):

- Normal
- Diabetes (Diabetic Retinopathy)
- Glaucoma
- Cataract
- AMD (Age-related Macular Degeneration)
- Hypertension
- Myopia
- Other abnormalities

## Dataset

| Aspect | Detail |
|:-------|:-------|
| Total images | 37,649 |
| Train / Val / Test split | 64% / 16% / 20% (24,095 / 6,024 / 7,530) |
| Sources | ODIR-5K (6,392 images) + two supplementary fundus datasets |
| Labels | Multi-label, 8 classes |

`full_df.csv` in this repo is the combined, cleaned label set used for training.

## Model & Training

- **Backbone:** DenseNet-121 (ImageNet-pretrained), `GlobalAveragePooling2D` → `Dense(512, ReLU)` + `Dropout` → `Dense(8, Sigmoid)`
- **Training strategy:** two-phase fine-tuning — Phase 1 trains the classification head with the DenseNet base frozen (5 epochs), Phase 2 unfreezes the base for end-to-end fine-tuning
- **Framework:** TensorFlow / Keras, trained on Kaggle (Tesla P100 GPU)

### Test Set Results

| Metric | Value |
|:-------|:-----:|
| AUC | 0.9666 |
| Binary Accuracy | 94.69% |
| Macro F1 | 0.82 |
| Test Loss | 0.150 |

### Per-Class Results (0.5 threshold)

| Disease | Precision | Recall | F1 | Support |
|:--------|:---------:|:------:|:--:|:-------:|
| Cataract | 0.91 | 0.90 | **0.91** | 928 |
| Myopia | 0.89 | 0.88 | **0.88** | 650 |
| AMD | 0.86 | 0.85 | **0.86** | 612 |
| Hypertension | 0.85 | 0.84 | **0.85** | 600 |
| Glaucoma | 0.81 | 0.85 | **0.83** | 1,139 |
| Normal | 0.78 | 0.86 | **0.82** | 1,857 |
| Diabetes | 0.84 | 0.71 | **0.77** | 1,756 |
| Other | 0.76 | 0.57 | **0.65** | 1,082 |

*(Metrics taken directly from the classification report in [final_best_model_notebook.ipynb](final_best_model_notebook.ipynb).)*

## Repository Contents

This repo holds the research/analysis artifacts, not a packaged application:

```
├── notebook.ipynb                                  # EDA, data cleaning & integration
├── final_best_model_notebook.ipynb                 # Model build, training, evaluation
├── final_best_model_notebook - Jupyter Notebook.pdf # PDF export of the above
├── full_df.csv                                      # Combined, cleaned label dataset
├── Final Report.pdf                                  # Full written report
├── Oculus_Presentation.pdf                           # Slide deck
└── WEB_REPORT/                                       # Submodule: project website source
```

The deployed Streamlit app and the project website are separate repos — see the Quick Links above.

## Known Limitations

- **"Other" class** has the weakest recall (0.57) — it groups heterogeneous pathologies that don't share consistent visual features.
- **Diabetes recall** (0.71) is the second lowest — the model misses some early-stage cases.
- Single modality: fundus images only, no OCT, visual field, or IOP data.
- Not a diagnostic device — intended as an assistive screening/triage tool, not a replacement for specialist review.

## Team

6-person Moringa School Data Science capstone team:

- Jeff Munyaka Mogaka — Project Lead
- Kitts Kikumu
- Kelvin Kinoti
- Judith Otieno
- Teresia Ndung'u
- Fridah Njung'e

## License

No license file is currently attached to this repository.
