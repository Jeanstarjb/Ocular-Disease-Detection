# Ocular-Disease-Detection
1. Business Understanding
1.1 Project Background: The Scalability Crisis in Ophthalmology
Ocular diseases such as Diabetic Retinopathy (DR), Glaucoma, and Cataracts are leading causes of preventable blindness worldwide. Diagnosis relies on manual examination of retinal fundus images by highly trained ophthalmologists.
This manual process faces critical challenges, driving the need for automation:
•	Scalability & Accessibility: A global shortage of ophthalmologists, especially in remote and underserved regions, creates severe bottlenecks and long wait times.
•	Time-Consuming: Specialists spend significant time reviewing normal, healthy scans, reducing their capacity for complex cases.
•	Human Factor: The process is subject to fatigue and inter-observer variability, risking missed findings.
1.2 Problem Statement & Objectives
The Problem Statement is that the current manual screening is inefficient, unscalable, and inaccessible to large parts of the population, leading to preventable vision loss.
This project addresses this need by developing a Clinical Decision Support System (CDSS) through a multi-label classification approach.
Key Objectives:
1.	Develop a Multi-Label Model: Build a deep learning model (DenseNet-121) that accurately detects 8 distinct ocular pathologies (Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other) from a single fundus image.
2.	Integrate Triage: Collaborate with as a triage assistant to flag high-risk images for immediate specialist review.
3.	Enhance Efficiency: Automate the screening of healthy/normal scans to reduce the manual review burden.
4.	Deploy an Accessible Tool: Deliver the trained model as a functional web application for easy clinical use.
________________________________________
2. Executive Summary
This project successfully developed a multi-label ocular disease classification system using deep learning on fundus images. The final DenseNet-121 model achieved a Macro F1-Score of 0.7871 and Test AUC of 0.9074, significantly outperforming the baseline CNN model (Macro F1: 0.5386, Test AUC: 0.8682).
The final model is capable of simultaneously detecting 8 distinct ocular pathologies from a single fundus image with clinically relevant accuracy.
Key Achievement
The model can simultaneously detect 8 distinct ocular pathologies from a single fundus image with clinically relevant accuracy.
________________________________________
3. Technical Approach
3.1 Final Model Architecture
The final system utilizes transfer learning with a pre-trained DenseNet-121 model. It was chosen because dense connections improve gradient flow and feature reuse.
•	Base Model: DenseNet-121, pre-trained on ImageNet.
•	Custom Classification Head: Added for 8-class multi-label output, consisting of GlobalAveragePooling2D, two Dropout(0.5) layers, and a final Dense(8, Sigmoid) layer.
•	Training Strategy: A two-phase training strategy was implemented:
1.	Phase 1 (Feature Extraction): Training only the classification head for 5 epochs with a learning rate of $1e-4$.
2.	Phase 2 (Fine-Tuning): Unfreezing the entire network (except BatchNorm layers) and training for 25 additional epochs with a lower learning rate ($1e-5$).
3.2 Dataset and Preparation
•	Total Images: 21,793 images, resulting from the integration of the original ODIR-5K dataset (6,392 images) with two external augmented datasets.
•	Classes: 8 disease classes (multi-label).
o	Classes: Normal (N), Diabetes (D), Glaucoma (G), Cataract (C), AMD (A), Hypertension (H), Myopia (M), and Other Abnormalities (O).
•	Class Imbalance: Significant class imbalance exists, with Normal (24.8%) and Diabetes (12.3%) dominating, and rare classes representing less than 2.5% each (Glaucoma, Cataract, AMD, Hypertension, Myopia).
•	Splitting: The combined dataset was split using a random seed (42) for reproducibility:
o	Training: 64% (13,945 images).
o	Validation: 16% (3,486 images).
o	Test: 20% (4,362 images).
________________________________________
4. Clinical Implications & Future Work
4.1 Recommended Use Cases
The primary goal is to develop a Clinical Decision Support System (CDSS) that serves as an automated first-pass screening tool. Recommended applications include:
•	Pre-screening Triage: Flag abnormal cases for urgent review.
•	Normal Scan Filtering: Automate healthy eye identification to enhance efficiency and reduce specialist workload.
•	Remote Screening: Enable diagnosis in underserved areas.
4.2 System Limitations & Future Work
Key limitations and recommendations for future improvements:
Limitation	Recommendation (Short-Term: 3-6 months)
"Other" Class Heterogeneity	Expand the "Other" Class by splitting it into specific sub-categories (e.g., Retinal Detachment, Macular Edema).
Black Box Nature	Implement Grad-CAM Visualization to provide heatmaps, showing which image regions influenced the predictions to build clinician trust.
No Severity Grading	Train ordinal classification for DR stages and AMD severity levels.
Validation Gap	Conduct External Validation by testing on public datasets like Messidor-2, EyePACS, and APTOS to assess generalization.
________________________________________
5. Technical Specifications
Software Dependencies
The core deep learning environment relies on the following major packages:
•	tensorflow==2.13.0
•	keras==2.13.0
•	numpy==1.24.3
•	scikit-learn==1.3.0
Inference Specifications
•	Recommended Hardware: NVIDIA GTX 1060 (6GB) GPU or higher for real-time inference.
•	Input Format: RGB image (JPEG/PNG), resized automatically to $224 \times 224$ pixels.
•	Model Weights File: densenet121_best_model_phase2.keras.weights.h5 (~230MB).
