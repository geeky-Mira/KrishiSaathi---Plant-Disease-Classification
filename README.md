# 🌿 KrishiSaathi – Plant Disease Detection App  

[![Live App](https://img.shields.io/badge/🌐_Try_App-Streamlit-success?style=for-the-badge)](https://krishisaathi.streamlit.app/)
[![View Notebook](https://img.shields.io/badge/Jupyter_Notebook-View-blue?style=for-the-badge)](https://github.com/geeky-Mira/KrishiSaathi---Plant-Disease-Classification/blob/main/plant-disease-detection.ipynb)
[![Main Script](https://img.shields.io/badge/Main_Script-Python-blueviolet?style=for-the-badge)](https://github.com/geeky-Mira/KrishiSaathi---Plant-Disease-Classification/blob/main/main.py)

---

### 🧠 Overview
**KrishiSaathi** is an AI-powered assistant that detects **plant leaf diseases** and provides **voice-enabled guidance** using **Google Gemini** and **Text-to-Speech (TTS)**.  
Built to support farmers and agriculture students, it blends deep learning, natural language understanding, and accessible UX.

> 🧾 Achieved **97% accuracy** on 87K+ leaf images using a CNN trained from scratch.

---

## 🚀 Key Features
- 🧬 **Smart Disease Detection:** CNN identifies leaf diseases across crops like tomato, potato, and corn.  
- 💬 **AI Assistant (Gemini):** Explains causes, treatments, and preventive measures in natural language.  
- 🔊 **Voice Feedback (TTS):** Converts diagnosis and tips into spoken output.  
- 🌐 **Streamlit Web App:** Lightweight, responsive, and easily deployable.  
- 📈 **High Performance:** 97% test accuracy on large multi-class dataset.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-------------|
| Deep Learning | **TensorFlow / Keras (CNN)** |
| Web Framework | **Streamlit** |
| AI Assistant | **Google Gemini API** |
| Speech Output | **Google Text-to-Speech (TTS)** |
| Dataset | [New Plant Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) |
| Language | Python |

---

## 📊 Model Insights
**Architecture:**  
Convolution → ReLU → MaxPooling → Dropout → Dense → Softmax  

**Metrics:**  
- ✅ **Training Accuracy:** 98.3%  
- ✅ **Validation Accuracy:** 96.8%  
- ✅ **Test Accuracy:** **97%**

**Input:** 224×224 RGB leaf image  
**Output:** Predicted disease label (e.g., *Tomato Early Blight*)  

Notebook → [🔗 plant-disease-detection.ipynb](https://github.com/geeky-Mira/KrishiSaathi---Plant-Disease-Classification/blob/main/plant-disease-detection.ipynb)

---

## 🧩 App Workflow

1. 📥 **Upload** or capture a leaf image  
2. 🧠 **CNN Model** predicts disease type  
3. 🤖 **Gemini** generates a detailed explanation  
4. 🔊 **TTS** speaks out diagnosis and remedy  
5. 🌍 **Streamlit UI** presents the complete result interactively  

Script → [🔗 main.py](https://github.com/geeky-Mira/KrishiSaathi---Plant-Disease-Classification/blob/main/main.py)

---

## 🌾 Dataset Details
- **Source:** [Kaggle – New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  
- **Samples:** 87,000+ images across 38 plant categories  
- **Processing:**  
  - Resized to 224×224  
  - Normalized pixel values  
  - Data augmentation (flip, rotation, zoom)  

---

## 🏁 Results Snapshot
| Metric | Score |
|--------|--------|
| Training Accuracy | 98.3% |
| Validation Accuracy | 96.8% |
| Test Accuracy | **97%** |
| Loss | 0.07 |

📊 Confusion matrix and training graphs available in the notebook.

---

## 🌱 Future Enhancements
- 📱 Mobile app using **TensorFlow Lite** for offline predictions  
- ☁️ Cloud API integration for IoT-based smart farming  
- 🧠 Explainable AI with Grad-CAM for disease localization  
- 🗣️ Multilingual voice support (Hindi, Bengali, etc.)  
- 💬 Improved Gemini conversational flow for personalized farming advice  

---

## 🧾 Project Summary 
**Objective:** AI-driven plant disease diagnosis and assistive communication system  
**Outcome:** 97% accurate CNN model + Streamlit deployment integrated with Google Gemini and TTS.  

---

## 📚 References
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Google Gemini API](https://ai.google.dev/)
- [Google Cloud Text-to-Speech](https://cloud.google.com/text-to-speech)
- [Kaggle Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

### 👨‍💻 Author
**Miratun Nahar**  
Building AI systems for language, agriculture, and real-world impact 🌍  
🔗 [GitHub – geeky-Mira](https://github.com/geeky-Mira)

