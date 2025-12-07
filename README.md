
## 📌 **AI Blackboard — Handwritten Digit Predictor (MNIST)**

A fun and interactive deep learning web app where users can **draw any digit (0–9)** on a virtual blackboard, and the system predicts it using a **CNN model trained on MNIST**.

Live Demo 👉 https://deeplearning-6gd8qpwx2gsdvpupa3iza5.streamlit.app/

---

## 🧠 **Project Overview**

This application allows users to:

* Draw digits using a canvas 🎨
* Convert drawing to grayscale 🖤
* Resize to 28×28 (MNIST format) 📏
* Predict the digit using a CNN 🔢

The project is built with:

| Purpose                 | Technology                |
| ----------------------- | ------------------------- |
| Web App                 | Streamlit                 |
| Drawing Canvas          | streamlit-drawable-canvas |
| Deep Learning Framework | PyTorch                   |
| Dataset                 | MNIST                     |
| Programming Language    | Python                    |

---

## 🚀 **Features**

✔ Fully browser-based — no installation needed
✔ Real-time digit prediction
✔ High confidence score display
✔ CPU-optimized — runs even without GPU
✔ Clean UI and smooth drawing experience

---

## 📂 **Repository Structure**

```
├─ mnist_cnn_model.pth       # Trained CNN model (required)
├─ third.py                  # Main Streamlit app
├─ requirements.txt          # Required dependencies
└─ README.md                 # Documentation
```

---

## 🔧 **Run Locally**

### 1️⃣ Clone the repo

```
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the application

```
streamlit run third.py
```

App will start here 👉 [http://localhost:8501](http://localhost:8501)

---

## 🏗 **Model Architecture (CNN)**

```
Input: 1×28×28
→ Conv2D (16 filters)
→ MaxPool
→ Conv2D (32 filters)
→ MaxPool
→ Flatten
→ FC (128 units)
→ FC (10 units)
Output: Digit class (0–9)
```

Activation: **ReLU**
Loss: **Cross Entropy**
Optimizer: **Adam**

---

## 📸 **Screenshots**

| Drawing               | Prediction            |
| --------------------- | --------------------- |
|<img width="1919" height="867" alt="image" src="https://github.com/user-attachments/assets/3d9ac058-bd96-44ca-b3e7-625572a3e167" />| <img width="1919" height="909" alt="image" src="https://github.com/user-attachments/assets/a4dd07fd-43e2-4ff6-a75f-31f815e850b7" />|

To add screenshots → upload PNG files in repo then paste image URLs here.

---

## 🌍 **Deploy on Streamlit Cloud**

1. Push code + `requirements.txt` to GitHub
2. Upload `mnist_cnn_model.pth` (IMPORTANT)
3. Streamlit Cloud → New App → Select Repository → Deploy
4. Enjoy 🚀

---

## 🤝 **Contributors**

| Name           | Role                                |
| -------------- | ----------------------------------- |
| Thanush Shetty | Developer — Model + UI + Deployment |
|thanushshetty7@gmail.com                              |

---

## ⭐ **Support**

If you like this project:

```
⭐ Star this repo
```

and share your feedback 💬

---
