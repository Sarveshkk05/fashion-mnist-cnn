# Fashion MNIST Classification with VGG16

This project demonstrates how to use transfer learning to classify clothing items from the Fashion MNIST dataset using a pretrained VGG16 model. Instead of training a deep network from scratch, the convolutional feature extractor from VGG16 is reused and a custom classifier is trained on top of it.

The primary goal is to understand the transfer learning workflow used in practical deep learning applications rather than focusing only on achieving the highest possible accuracy.

---

## Overview

In this project, the following steps are performed:

* Loading the Fashion MNIST dataset from a CSV file
* Converting grayscale images into a format compatible with VGG16
* Freezing pretrained convolutional layers
* Training a custom fully connected classifier
* Evaluating performance on unseen data

Since VGG16 expects 3-channel 224×224 images, the original 28×28 grayscale images are preprocessed accordingly.

---

## Dataset

Fashion MNIST contains 60,000 grayscale images across 10 clothing categories.

| Label | Class       |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

Download the dataset CSV from Kaggle and place it in a suitable location on your system. Update the dataset path inside the code if necessary.

---

## Requirements

```bash
pip install torch torchvision numpy pandas pillow scikit-learn matplotlib
```

The code automatically uses a GPU if available; otherwise, it falls back to CPU.

---

## Project Structure

```
fashion-label_using_cnn/
├── cnn-versions/
│   ├── model_v1.py
│   └── model_v2.py
├── README.md
└── .gitignore
```

* `cnn-versions/model_v1.py` — initial implementation of the model and training pipeline
* `cnn-versions/model_v2.py` — improved or modified version of the model
* `README.md` — project documentation
* `.gitignore` — specifies files excluded from version control

---

## Pipeline Description

### Data Loading

* Column 0 represents labels
* Columns 1–784 represent pixel values

---

### Train–Test Split

* 80% training
* 20% testing

This split allows evaluation of model generalization.

---

### Image Preprocessing

Each 28×28 grayscale image undergoes the following steps:

1. Convert grayscale to 3-channel RGB by duplicating the single channel
2. Resize to 256×256
3. Center crop to 224×224
4. Convert to tensor using `ToTensor()` (scales pixel values to [0, 1])
5. Normalize using ImageNet statistics

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

`ToTensor()` must be applied before `Normalize()` because normalization operates on tensors.

---

### Model Architecture

A pretrained VGG16 network is used as a frozen feature extractor. Only the classifier head is replaced and trained.

Classifier structure:

```
Linear(25088 → 1024)
ReLU
Dropout(0.5)

Linear(1024 → 512)
ReLU
Dropout(0.5)

Linear(512 → 10)
```

---

### Training Setup

* Optimizer: Adam
* Learning rate: 0.0001
* Loss function: CrossEntropyLoss
* Batch size: 32
* Epochs: 10

Only the classifier parameters are updated during training.

---

### Evaluation

Training and test accuracy are computed to assess performance and detect potential overfitting or underfitting.

---

## How to Run

Navigate to the folder containing the model version you want to execute and run the script.

Example:

```bash
python cnn-versions/model_v1.py
```

or

```bash
python cnn-versions/model_v2.py
```

---

## Configuration Summary

| Parameter        | Value            |
| ---------------- | ---------------- |
| Batch size       | 32               |
| Learning rate    | 0.0001           |
| Epochs           | 10               |
| Optimizer        | Adam             |
| Loss             | CrossEntropyLoss |
| Train/Test split | 80% / 20%        |

---

## Performance Considerations

It is possible for a simple custom CNN to outperform VGG16 on this dataset.

Fashion MNIST images are small and relatively low in complexity. VGG16 is significantly larger than necessary, and resizing images from 28×28 to 224×224 introduces interpolation artifacts that may slightly affect performance.

The main value of this project lies in understanding:

* Transfer learning workflows
* Preprocessing for pretrained models
* PyTorch training pipelines
* Model comparison approaches

---

## Learning Outcomes

By completing this project, you will gain experience with:

* Transfer learning concepts
* Freezing versus fine-tuning layers
* Image preprocessing for pretrained networks
* Training and evaluation pipelines
* Model comparison strategies

These skills are directly applicable to practical deep learning projects.
