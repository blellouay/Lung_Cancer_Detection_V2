# Lung Cancer CT Scan Classification Platform

## Overview

This project is an end-to-end deep learning platform for automated lung cancer subtype classification from CT scan images.

It was designed to classify lung CT scans into four categories:

* Adenocarcinoma
* Large Cell Carcinoma
* Squamous Cell Carcinoma
* Normal

The platform combines:

* Multiple scratch CNN architectures
* Transfer learning experiments
* Advanced class balancing strategies
* Medical interpretability with Grad-CAM
* Deployment-ready Streamlit dashboard
* Dynamic model loading and prediction demo

---

# Project Goals

## Primary objectives:

* Improve lung cancer subtype classification accuracy
* Address class imbalance and minority subtype collapse
* Build interpretable medical AI models
* Compare multiple architectures fairly
* Deploy the best models through an interactive dashboard

---

# Key Features

## Data Pipeline

* Structured train / validation / test loaders
* Custom PyTorch dataset
* Medical image preprocessing
* Image augmentation
* ImageNet normalization

---

## Models Implemented

### Scratch Models

* CNNBaseline
* ResNet18Scratch
* VGG16Scratch
* MobileNetV2Scratch
* EfficientNetB0Scratch

### Transfer Learning Models

* ResNet18Transfer
* VGG16Transfer
* MobileNetV2Transfer
* EfficientNetB0Transfer
* InceptionV3Transfer

---

## Advanced Training Strategies

* WeightedRandomSampler
* Focal Loss
* Early stopping
* Learning rate scheduling
* Weight decay
* Fine-tuning for transfer models
* Deterministic reproducible training

---

## Evaluation System

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1
* Confusion Matrix
* Per-Class Recall
* Minimum Class Recall
* Weakest Class Detection

---

## Interpretability

* Grad-CAM heatmaps
* Automatic visual explanations
* Medical deployment transparency

---

# Dashboard Features

The Streamlit dashboard includes:

* Model comparison
* Best model ranking
* Per-class recall analysis
* Training and validation curves
* Architecture visualization
* Deployment readiness checks
* Prediction demo
* Confidence scores
* Grad-CAM visualization
* Dynamic model loading

---

# Final Best Models

## Best Scratch Model

```text
ResNet18Scratch
```

## Best Transfer Learning Model

```text
ResNet18Transfer
```

---

# Final Training Recipe

```text
WeightedRandomSampler
FocalLoss(gamma=1.0)
Adam Optimizer
Weight Decay
StepLR Scheduler
Early Stopping
Grad-CAM
```

---

# Project Architecture

```text
project/
├── data/
├── src/
│   ├── data/
│   ├── model/
│   ├── training/
│   ├── evaluation/
│   ├── loss/
│   ├── interpretability/
│   ├── Save/
│   ├── plot/
│   └── deployment/
├── results/
└── dash/
```

---

# Main Deliverables

* Professional medical AI pipeline
* Multi-model experimentation framework
* Transfer learning benchmark
* Explainable AI integration
* Streamlit deployment dashboard
* Prediction demo
* Training analytics
* Deployment-ready saved models

---

# Future Improvements

* Two-stage classification pipeline:

  * Normal vs Abnormal
  * Cancer subtype classification
* Additional hyperparameter optimization
* Ensemble models
* Expanded dataset support
* ONNX deployment optimization

---

# Conclusion

This project evolved from a simple CNN classifier into a professional, interpretable, and deployable medical AI platform focused on robust lung cancer subtype classification.

It demonstrates:

* Deep learning experimentation
* Medical AI deployment
* Explainable AI
* Dashboard engineering
* Transfer learning benchmarking

---

# Author

Developed as a complete medical imaging AI research and deployment project.
