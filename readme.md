# Deepfake Detection for Human Face Images

## 📌 Project Overview

Deepfake technology poses serious threats to digital authenticity, privacy, and security by enabling the creation of highly realistic manipulated images and videos. This project focuses on **detecting deepfake human face images** using **deep learning techniques**, primarily **Convolutional Neural Networks (CNNs)**, to distinguish between real and manipulated content.

The system integrates:

* A **TensorFlow-based deep learning model** for detection
* A **FastAPI backend** for real-time inference
* A **ReactJS frontend** for user interaction

The goal is to provide an **accurate, scalable, and user-friendly deepfake detection solution**.

---

## 🎯 Objectives

* Detect real vs fake human face images with high accuracy
* Strengthen trust in digital media and online content
* Support applications in digital forensics and media verification
* Provide real-time predictions through a web-based interface
* Address ethical and security concerns arising from deepfake misuse

---

## 🧠 Methodology

1. **Dataset Preparation**

   * Real and deepfake face images collected from public datasets (Kaggle)
   * Images resized, normalized, and augmented (flip, rotation, noise)

2. **Preprocessing**

   * Face detection using OpenCV
   * Image normalization to improve training stability

3. **Model Training**

   * CNN-based architecture for spatial feature extraction
   * Binary classification (Real / Fake)
   * Evaluation using Accuracy, Precision, Recall, F1-score, and AUC

4. **Deployment Pipeline**

   * Trained model exported for inference
   * TensorFlow Serving for scalable predictions
   * FastAPI backend communicates with the model
   * ReactJS frontend for image upload and result display

---

## 🏗️ System Architecture

```
User → ReactJS UI → FastAPI Backend → TensorFlow Serving → CNN Model → Result
```

---

## 🧪 Algorithms Used

* **Custom CNN** for image-based deepfake detection
* **ResNet50** (evaluated, limited performance)
* **MobileNetV2** (best-performing model with high accuracy and efficiency)

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* AUC (ROC Curve)

MobileNetV2 achieved the best overall performance while maintaining low computational cost, making it suitable for real-world deployment.

---

## 🚧 Limitations

* Limited generalization to unseen deepfake generation techniques
* Performance depends on input image quality
* High computational requirements for deep CNN models
* Susceptibility to highly sophisticated GAN-based deepfakes

---

## 🔮 Future Scope

* Use advanced architectures (EfficientNet, Transformers)
* Expand and diversify datasets
* Extend detection to video deepfakes using temporal modeling
* Improve robustness against adversarial attacks
* Deploy on edge and mobile platforms
* Add explainability and visualization of manipulated regions

---

## 👥 Team & Contributions

This was a **team-based academic project** carried out as part of the B.Tech program.

**Team Members:**

* Bestin K Benny
* **Arjun Vasavan**
* E N Aadhila Nazeer
* Hena Maria Biju
* Jervin Abraham Kuriakose

**My Contribution (Arjun Vasavan):**

* Deep learning model design and training
* Dataset preprocessing and augmentation
* CNN-based deepfake detection pipeline
* Model evaluation and optimization

Backend API and frontend UI were developed collaboratively by the team.

---

## 📁 Repository Scope

> ⚠️ **Note**

* Datasets and trained model weights are **not included** due to size constraints.
* This repository focuses on the **machine learning training and inference pipeline**.
* The model can be reproduced using the provided training code.

---

## 🛠️ Technologies Used

* Python 3.x
* TensorFlow / Keras
* OpenCV
* FastAPI
* ReactJS
* NumPy, Matplotlib
* Git & GitHub

---

## 📜 Academic Context

This project was submitted to **APJ Abdul Kalam Technological University**
as part of the **B.Tech (ECE) with Minor in Computer Science** program
at **College of Engineering Chengannur** (November 2024)

---

## 📄 License

This project is for academic and educational purposes.

---

## 🤝 Acknowledgments

We would like to thank our faculty advisors and the College of Engineering Chengannur for their guidance and support throughout this project.
