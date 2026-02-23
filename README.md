# Fashion MNIST Classification with VGG16 👕👟

This project demonstrates how to use **transfer learning** to classify clothing items from the Fashion MNIST dataset using a pretrained VGG16 model. Instead of training a deep network from scratch, we reuse VGG16's feature extraction capability and train a custom classifier on top.

The main objective is to learn the **transfer learning workflow used in real-world deep learning projects**, not just to achieve the highest accuracy.

---

## 📌 Overview

In this project, we:

* Load the Fashion MNIST dataset from a CSV file
* Convert grayscale images into a format compatible with VGG16
* Freeze pretrained convolutional layers
* Train a custom fully connected classifier
* Evaluate performance on unseen data

Since VGG16 expects **3-channel 224×224 images**, we preprocess the original 28×28 grayscale images accordingly.

---

## 📊 Dataset

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

Download the dataset CSV from [Kaggle — Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) and place it inside your project folder:

```
project_root/
├── data/
│   └── fashion-mnist_train.csv
└── CNN.ipynb
```

Using a **relative path** instead of a hardcoded absolute path makes the project easier to share and run on different systems.

Example:

```python
import os

DATASET_PATH = os.path.join("data", "fashion-mnist_train.csv")
```

---

## ⚙️ Requirements

```bash
pip install torch torchvision numpy pandas pillow scikit-learn matplotlib
```

The code automatically uses GPU if available, otherwise falls back to CPU.

---

## 📁 Project Structure

```
fashion-label_using_cnn/
├── cnn-versions/
│   ├── model_v1/
│   └── model_v2/
├── data/
│   └── fashion-mnist_train.csv
├── CNN.ipynb
└── .gitignore
```

* `cnn-versions/model_v1/` — first version of the CNN model
* `cnn-versions/model_v2/` — updated/improved version of the CNN model
* `data/` — contains the Fashion MNIST CSV dataset (not tracked by Git)
* `CNN.ipynb` — main notebook with the full training pipeline
* `.gitignore` — specifies files and folders excluded from version control

---

## 🔍 How the Pipeline Works

### 1️⃣ Data Loading

* Column 0 → labels
* Columns 1–784 → pixel values

---

### 2️⃣ Train-Test Split

* 80% training
* 20% testing

This allows us to evaluate generalization performance.

---

### 3️⃣ Image Preprocessing

Each 28×28 grayscale image is processed as follows:

1. Convert grayscale → 3-channel RGB (duplicate channel 3 times)
2. Resize to 256×256
3. Center crop to 224×224
4. Convert to tensor using `ToTensor()` (scales pixel values to [0, 1])
5. Normalize using ImageNet statistics

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

⚠️ Note: `ToTensor()` must be applied before `Normalize()` since normalization operates on tensors, not PIL images.

---

### 4️⃣ Model Architecture

We use pretrained VGG16 as a **frozen feature extractor**.

Only the classifier head is replaced and trained:

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

### 5️⃣ Training Setup

* Optimizer: Adam
* Learning rate: 0.0001
* Loss: CrossEntropyLoss
* Batch size: 32
* Epochs: 10

Only classifier parameters are updated during training.

---

### 6️⃣ Evaluation

We compute:

* Training accuracy
* Test accuracy

This helps identify overfitting or underfitting.

---

## ▶️ How to Run

1. Download `fashion-mnist_train.csv` from [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist) and place it in the `data/` folder.
2. Open `CNN.ipynb` in Jupyter or Google Colab.
3. Update the `DATASET_PATH` variable if needed (or switch to a relative path as shown above).
4. Run all cells sequentially.

> **Expected runtime:** ~15–30 minutes on CPU, ~3–5 minutes on a GPU, for 10 epochs.  
> **Expected accuracy:** ~88–91% on training data, ~85–88% on test data.

---

## 🔧 Configuration Summary

| Parameter        | Value            |
| ---------------- | ---------------- |
| Batch size       | 32               |
| Learning rate    | 0.0001           |
| Epochs           | 10               |
| Optimizer        | Adam             |
| Loss             | CrossEntropyLoss |
| Train/Test split | 80% / 20%        |

---

## 💡 Performance Note (Important)

Do not be surprised if a **simple custom CNN** outperforms VGG16 on this dataset.

Fashion MNIST images are:

* Very small (28×28)
* Low complexity
* Limited variation

VGG16 is significantly larger than necessary, so it may not provide major accuracy gains. The upscaling from 28×28 to 224×224 also introduces interpolation artifacts that can slightly hurt performance.

The real value of this project is:

✅ Understanding transfer learning workflow  
✅ Learning preprocessing for pretrained models  
✅ Practicing PyTorch training pipelines  
✅ Comparing architectures

---

## 🚀 Learning Outcomes

By completing this project, you will understand:

* Transfer learning concepts
* Freezing vs fine-tuning layers
* Image preprocessing for pretrained networks
* Training and evaluation pipelines
* Model comparison strategies

These skills are directly applicable to industry deep learning tasks.

---
