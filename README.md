# Animal Image Classification with Deep Learning

A deep learning project that classifies animals from images using convolutional neural networks (CNNs), built as part of my Artificial Intelligence module at Goldsmiths, University of London. The project follows the universal deep learning workflow described in François Chollet's *Deep Learning with Python*.

---

## What it does

Trains and compares multiple CNN architectures to classify images into 6 animal categories, exploring how different regularisation techniques and transfer learning affect model performance and generalisation.

**Classes:** Bird · Cat · Deer · Dog · Frog · Horse  
**Dataset:** CIFAR-10 (filtered to animal classes only — 30,000 training images, 6,000 test images)

---

## Models Compared

| Model | Validation Accuracy |
|-------|-------------------|
| Baseline CNN (no regularisation) | 67.5% |
| CNN + Dropout | 70.9% |
| CNN + Data Augmentation | 67.2% |
| Transfer Learning (MobileNetV2) | **76.2%** |

**Final test accuracy: 77%** (Transfer Learning model)

---

## Key Findings

- The baseline model heavily overfitted — training accuracy reached 86% vs 67.5% validation, a gap of ~19%
- Dropout reduced the overfitting gap from 19% to just 2.7%, proving more effective than data augmentation in this case
- Data augmentation underperformed on CIFAR-10's 32×32 images — at such low resolution, rotation and zoom degrade useful pixel information rather than adding meaningful variety
- MobileNetV2 pretrained on ImageNet significantly outperformed all models trained from scratch, demonstrating the power of transfer learning on small datasets
- Hyperparameter search found learning rate 0.001 to be optimal — 0.01 was too unstable, 0.0001 too slow to converge
- **Cat** was the hardest class to classify (65% recall), frequently confused with dog
- **Frog** was the easiest (92% recall), likely due to its distinctive green colouring

---

## Experiments

### Regularisation
- **Experiment A:** Added Dropout (0.5 after conv layers, 0.3 after dense)
- **Experiment B:** Data augmentation (random horizontal flip, rotation ±10%, zoom ±10%)

### Hyperparameter Search
Tested learning rates: `[0.01, 0.001, 0.0001]` — evaluated by best validation accuracy over 15 epochs

### Transfer Learning
Fine-tuned MobileNetV2 (pretrained on ImageNet, frozen base) with a custom classification head for 6 animal classes

### Model Interpretability
Implemented **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualise which regions of each image the model attended to when making predictions — exploring interpretability beyond standard accuracy metrics

---

## Pipeline

1. Load CIFAR-10 and filter to 6 animal classes
2. Remap class labels (0–5)
3. Check class balance
4. Normalise pixel values (0–255 → 0–1)
5. Build and train baseline CNN
6. Apply regularisation experiments (Dropout, Augmentation)
7. Hyperparameter search across learning rates
8. Transfer learning with MobileNetV2
9. Final evaluation on held-out test set
10. Grad-CAM visualisation
11. Error analysis — correct vs incorrect predictions

---

## How to Run

1. Clone the repo and open the notebook in Jupyter
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run all cells top to bottom — CIFAR-10 downloads automatically via Keras

> **Note:** Training the transfer learning model is faster with a GPU. On CPU it will still run but may take longer.

---

## Requirements

```
tensorflow
numpy
matplotlib
scikit-learn
```

---

## References

- Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. *(CIFAR-10 dataset)*
- Sandler et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.

---

## Technologies

Python · TensorFlow · Keras · NumPy · Matplotlib · Scikit-learn · Jupyter Notebook
