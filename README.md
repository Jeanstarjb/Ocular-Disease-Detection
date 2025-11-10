# 👁️ Ocular Disease Multi-Label Classification using DenseNet-121

## 1. Business Understanding

### 1.1 Project Background: The Scalability Crisis in Ophthalmology

Ocular diseases such as **Diabetic Retinopathy (DR)**, **Glaucoma**, and **Cataracts** are leading causes of preventable blindness worldwide.  
Diagnosis relies on **manual examination of retinal fundus images** by highly trained ophthalmologists — a process that faces critical challenges, driving the need for automation.

#### Key Challenges
- **Scalability & Accessibility:**  
  Global shortage of ophthalmologists, especially in remote regions, causes severe bottlenecks and long wait times.
- **Time-Consuming:**  
  Specialists spend significant time reviewing normal scans, reducing capacity for complex cases.
- **Human Factor:**  
  Subject to fatigue and inter-observer variability, risking missed findings.

---

### 1.2 Problem Statement & Objectives

#### Problem Statement
The current manual screening process is **inefficient, unscalable, and inaccessible**, leading to preventable vision loss.  
This project develops a **Clinical Decision Support System (CDSS)** using a **multi-label classification approach**.

#### Objectives
- **Develop a Multi-Label Model:**  
  Build a **DenseNet-121** model to detect **8 ocular pathologies** from a single fundus image:
  - Normal  
  - Diabetes  
  - Glaucoma  
  - Cataract  
  - AMD  
  - Hypertension  
  - Myopia  
  - Other Abnormalities  
- **Integrate Triage:**  
  Assist as a triage tool to flag high-risk images for immediate specialist review.
- **Enhance Efficiency:**  
  Automate the screening of healthy scans to reduce manual workload.
- **Deploy an Accessible Tool:**  
  Deliver the trained model as a **web application** for clinical usability.

---

## 2. Executive Summary

This project successfully developed a **multi-label ocular disease classification system** using deep learning on fundus images.

| Metric | DenseNet-121 | Baseline CNN |
|:-------|:-------------:|:-------------:|
| **Macro F1-Score** | **0.7871** | 0.5386 |
| **Test AUC** | **0.9074** | 0.8682 |

✅ The model can detect **8 ocular pathologies simultaneously** from a single fundus image with clinically relevant accuracy.

---

## 3. Technical Approach

### 3.1 Final Model Architecture

The system employs **transfer learning** with a pre-trained **DenseNet-121** backbone for efficient feature reuse and gradient flow.

#### Architecture Summary
- **Base Model:** DenseNet-121 (pre-trained on ImageNet)  
- **Custom Classification Head:**
  - `GlobalAveragePooling2D`
  - Two `Dropout(0.5)` layers
  - `Dense(8, activation='sigmoid')` (multi-label output)

#### Training Strategy
- **Phase 1 – Feature Extraction:**  
  Train only the classification head for **5 epochs** (LR = `1e-4`).
- **Phase 2 – Fine-Tuning:**  
  Unfreeze the full network (except BatchNorm layers) and train for **25 epochs** (LR = `1e-5`).

---

### 3.2 Dataset & Preparation

- **Total Images:** 21,793  
  (ODIR-5K + 2 external augmented datasets)
- **Classes (8):**  
  N, D, G, C, A, H, M, O (Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other)
- **Class Imbalance:**  
  - Normal (24.8%) and Diabetes (12.3%) dominant  
  - Rare classes < 2.5%

#### Dataset Split
| Subset | Percentage | Image Count |
|:--------|:-----------:|------------:|
| Training | 64% | 13,945 |
| Validation | 16% | 3,486 |
| Test | 20% | 4,362 |

> **Random Seed:** 42 (for reproducibility)

---

## 4. Clinical Implications & Future Work

### 4.1 Recommended Use Cases
The CDSS serves as an automated **first-pass screening tool**.

- **Pre-Screening Triage:** Flag abnormal cases for urgent review.  
- **Normal Scan Filtering:** Automate identification of healthy eyes.  
- **Remote Screening:** Enable diagnosis in underserved areas.

---

### 4.2 System Limitations & Future Improvements

| Limitation | Recommendation (Short-Term: 3–6 months) |
|:------------|:----------------------------------------|
| “Other” class heterogeneity | Expand into sub-categories (e.g., Retinal Detachment, Macular Edema). |
| Black-box nature | Implement **Grad-CAM** visualizations for transparency. |
| No severity grading | Train **ordinal models** for DR and AMD stages. |
| Validation gap | Test externally on **Messidor-2**, **EyePACS**, and **APTOS** datasets. |

---

## 5. Technical Specifications

### 5.1 Software Dependencies
```bash
tensorflow==2.13.0
keras==2.13.0
numpy==1.24.3
scikit-learn==1.3.0

5.2 Inference Specifications

Recommended GPU: NVIDIA GTX 1060 (6GB) or higher

Input Format: RGB (JPEG/PNG), resized to 224×224 px

Model Weights: densenet121_best_model_phase2.keras.weights.h5 (~230MB)

6. Conclusion

This project demonstrates how deep learning–based clinical decision support can tackle scalability, accessibility, and efficiency challenges in ophthalmology.
By automating the detection of multiple eye diseases from a single image, this system offers a scalable, deployable AI solution for early screening and vision preservation.

🧩 Repository Structure
├── data/
│   ├── train/
│   ├── test/
│   └── val/
├── models/
│   └── densenet121_best_model_phase2.keras.weights.h5
├── app/
│   └── streamlit_app.py
├── notebooks/
│   └── training_pipeline.ipynb
├── requirements.txt
└── README.md

🚀 How to Run

Clone the Repository

git clone https://github.com/yourusername/ocular-disease-detection.git
cd ocular-disease-detection


Install Dependencies

pip install -r requirements.txt


Run the Streamlit App

streamlit run app/streamlit_app.py


Upload a Fundus Image
The app automatically resizes the image to 224×224 px and outputs predicted pathologies with confidence scores.

🧠 Key Takeaways

The model achieves Macro F1 = 0.7871 and Test AUC = 0.9074

Built using DenseNet-121 with transfer learning

Deployed as a Streamlit-based web application

Supports multi-label disease detection for 8 ocular pathologies

📄 License

This project is released under the MIT License — free for research and educational use.
