# Draa (درء) – Violence Detection System Using Computer Vision

## 📌 Project Overview
Draa (درء) is a Computer Vision–based project for detecting violent activities in CCTV surveillance videos.  
The system analyzes video footage and classifies it into Normal or Violence, supporting public safety and smart city applications.

---

## 🎯 Project Goal
To detect violent behavior in CCTV videos using Computer Vision and Deep Learning techniques.

---

## 📂 Dataset
This project uses the Smart City CCTV Violence Detection Dataset (SCVD) from Kaggle.

Dataset link:  
https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd

Dataset type:
- CCTV video clips
- Real-world surveillance data

---

## 🏷️ Classes
The model performs binary classification with the following classes:
- Normal
- Violence

---

## 🧠 Model
Multiple deep learning models were experimented with.  
The final model will be specified after evaluation:
The approach combines:
- Computer Vision for spatial feature extraction
- Deep Learning for pattern recognition in video data

---

## 🛠️ Tools & Technologies
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Kaggle Notebook  
- FastAPI  

---

## 🏋️ Training Process
- Video frames are extracted and preprocessed
- Computer Vision techniques are applied to analyze visual features
- The model is trained to distinguish between violent and normal behavior
- Regularization techniques are used to reduce overfitting

---

## 📊 Evaluation
The model is evaluated using:
- Confusion Matrix
- Accuracy, Precision, and Recall

---

## 🚀 Deployment
The project can be executed on:
- Kaggle (model training and testing)
- FastAPI (model inference via an API)

---

## ▶️ How to Run
1. Open the notebook on Kaggle.
2. Add the dataset as an input.
3. Run all cells to train or load the model.
4. Use the FastAPI interface for inference if deployed.
