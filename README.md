# Civic Issue Image Classification 🚦

This project is a deep learning based image classification system that identifies civic infrastructure issues from images.  
It classifies images into the following categories:

- 🗑 Garbage
- ⚡ Electrical
- 🛣 Road

If the image does not clearly belong to any of these categories, it can be treated as **Other** using a confidence threshold during prediction.

---

## 📁 Dataset Structure
classification/
├── garbage/
├── electrical/
└── road/

Each folder contains images related to the respective category.

---

## 🧠 Model & Approach
- Transfer Learning using **MobileNetV2**
- Image size: **224 × 224**
- Softmax output layer with 3 classes
- Confidence threshold used to handle unknown images

---

## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
