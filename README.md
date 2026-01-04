# Self-Supervised Contrastive Learning on Longitudinal Physiological Data

## 📌 Project Overview
This project applies **SimCLR-style contrastive learning** to a four-year longitudinal dataset of physiological features (collected via Oura Ring) to detect health anomalies without relying on supervised labels.

The goal was to learn embeddings that capture meaningful variation in daily physiological states. Using a lightweight MLP encoder and a custom augmentation pipeline, the model successfully differentiates "sick" days from "healthy" days in a purely self-supervised manner.

## 🔍 Key Findings
* **Latent Space Clustering:** Unsupervised training revealed a distinct manifold containing approximately **70% of labeled "sick" days**.
* **Physiological Signals:** "Well-clustered" sick days were characterized by significant deviations in **Body Temperature Trends** and reduced **Metabolic Activity (MET)**.
* **Method Efficacy:** Proves that contrastive learning can extract discriminative health features from single-subject wearable data even with sparse positive labels.

## 🛠 Methodology

### 1. Data Preprocessing
* **Source:** 4 years of daily summaries from Oura Ring (Sleep, Readiness, Activity).
* **Feature Engineering:** 27 features selected (e.g., HRV, Temperature Deviation, Respiratory Rate); data was Z-scored and missing values dropped.

### 2. The SimCLR Framework
To learn representations without labels, the model generates positive pairs via data augmentation:
* **Augmentation Pipeline:** * Additive Gaussian noise to mimic sensor noise.
  * Random Feature Dropout (10%).
* **Encoder Architecture:** A simple 2-layer MLP (Linear 256 → ReLU) outputting 128-dimensional representations.
* **Loss Function:** NT-Xent loss with **Cosine Similarity** ($T=0.1$). 
  * *Note:* Cosine similarity was chosen over Euclidean distance to better capture the rhythmic/angular nature of physiological signals.

### 3. Training Configuration
* **Optimizer:** Adam
* **Epochs:** 100
* **Batch Size:** 64
* **Visualization:** UMAP used to project 128-dim embeddings into 2D space for analysis.

## 📊 Results Visualization

### UMAP Projection
*The model learned to cluster sick days (red) away from the baseline healthy state (blue).*
*(See `SimCLR_presentation.pdf` for full charts)*

### Discriminative Features
Analysis of the clusters showed that the model picked up on:
* **Temperature Deviation:** High variance in body temp during sick days.
* **Activity Burn / MET:** Sharp decrease in metabolic output.

## 🚀 Future Work
* **Early Warning Systems:** Investigating if embedding shifts occur *prior* to symptom onset.
* **Metric Stability:** Exploring Mahalanobis distance to better account for feature covariance, provided matrix stability issues are resolved.
* **Feature Pruning:** Removing highly correlated inputs to improve contrastive efficiency.

## 💻 Tech Stack
* **Language:** Python
* **Libraries:** PyTorch, Pandas, NumPy, UMAP-learn, Matplotlib/Seaborn

## 📂 Repository Structure
* `SimCLR_Analysis.ipynb`: Main notebook containing data loading, model training, and evaluation.
* `SimCLR_presentation.pdf`: Slide deck summarizing the theoretical background and visual results.
