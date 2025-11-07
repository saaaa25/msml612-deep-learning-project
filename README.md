# Multimodal Transformers for Sentiment & Emotion Recognition

This repository contains the code for our course project:

> **Multimodal Transformers for Sentiment & Emotion Recognition**  
> Team: Abhay Sagoti, Akshay Suresh, Satwika Konda, Sivani Mallangi, Srutileka Suresh  
> Course: Deep Learning – Final Project (Fall 2025)

We build a **multimodal transformer** that fuses **text, audio, and visual** streams via **cross-attention** for:
- **Sentiment regression** in \[-3, 3\]
- **3-class sentiment classification** (negative / neutral / positive)
- **6-class emotion recognition** (happy, sad, angry, fear, disgust, surprise)

---

##  Motivation

Human affect is **inherently multimodal**. Text alone often fails for:
- **Sarcasm** – “Great job.” can be either positive or negative.
- **Ambiguous tone** – resolved only via **prosody** (pitch, energy) and **facial cues** (AUs, micro-expressions).

Multimodal Transformers allow us to:
- Attend over **aligned sequences** from each modality
- Learn **cross-modal dependencies**
- Build **interpretable** models with attention maps and modality attribution

Applications include:
- Customer support analytics  
- Mental health screening assistance  
- Affect-aware tutoring systems  
- Human–robot interaction  

---

##  Problem Definition

Given aligned text tokens \(X_T\), audio frames \(X_A\), and visual frames \(X_V\):

\[
f(X_T, X_A, X_V) \rightarrow y
\]

where \(y\) includes:
- **Sentiment (regression)**: score in \[-3, 3\]  
- **Sentiment (3-class)**: \{-1, 0, +1\}  
- **Emotion (6-class)**: {happy, sad, angry, fear, disgust, surprise}

---

##  Datasets

**Primary dataset**
- **CMU-MOSEI**
  - ~23k labeled video segments
  - Modalities: text, audio, vision
  - Labels: sentiment \[-3, 3\] and Ekman emotions

**Secondary (optional)**
- **MELD** – multi-party dialogues (Friends TV show)
- **IEMOCAP** – dyadic acted emotions
- **CREMA-D** – controlled audio-visual emotion expressions

>  Due to dataset size, **raw datasets are not included** in this repo. Refer : https://www.kaggle.com/datasets/samarwarsi/cmu-mosei


---

##  Methodology

### 1. Modality-Specific Encoders
- **Text**: BERT / RoBERTa  
  Output: contextual embeddings \(E_T \in \mathbb{R}^{n \times d_T}\)
- **Audio**: wav2vec 2.0 or MFCC + prosody  
  Output: \(E_A \in \mathbb{R}^{m \times d_A}\)
- **Vision**: ResNet / ViT / OpenFace  
  Output: \(E_V \in \mathbb{R}^{k \times d_V}\)

### 2. Temporal Alignment
- Resample modalities to a common frame rate
- Add positional and modality-type embeddings

### 3. Fusion via Cross-Attention
Text serves as **query**, audio and vision as **key-value pairs**:

\[
Z_{\text{fusion}} = \text{MHA}(Q = Z_T, K = [Z_A; Z_V], V = [Z_A; Z_V])
\]

Stack **L layers** with residual connections and layer normalization.

### 4. Prediction Heads & Losses
- **Sentiment regression** – MSE  
- **3-class sentiment** – Cross-entropy  
- **Emotion classification** – Multi-label BCE or CE  

Total loss:

\[
L = \lambda_{\text{reg}} L_{\text{MSE}} + \lambda_{\text{cls}} L_{\text{CE}} + \lambda_{\text{emo}} L_{\text{BCE}}
\]

---

##  Implementation Details

- **Frameworks:** PyTorch, Hugging Face Transformers, Librosa, OpenFace, scikit-learn  
- **Training:** AdamW optimizer, cosine LR schedule, warmup, mixed precision  
- **Regularization:** Modality dropout, stochastic depth, SpecAugment  
- **Hardware:** Single GPU (RTX 4070 recommended)  

---

##  Evaluation

### Metrics
- **Regression:** MAE, RMSE, Pearson correlation  
- **Classification:** Accuracy, F1-score (macro), per-class recall  

### Baselines & Ablations
- Text-only BERT baseline  
- Audio-only wav2vec 2.0  
- Vision-only ResNet  
- Early vs. late fusion comparisons  
- Cross-attention ablations  

### Interpretability
- Token-level attention heatmaps  
- Modality attribution visualization  
- Grad-CAM on vision stream  

---
