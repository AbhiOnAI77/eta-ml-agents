# 🚴 Food Delivery ETA Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Used-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Used-yellow)

Welcome to the **Food Delivery ETA (Estimated Time of Arrival) Prediction System**! This interactive Streamlit application demonstrates a complete end-to-end Machine Learning and Deep Learning pipeline. It generates realistic synthetic data, performs comprehensive exploratory data analysis (EDA), and builds predictive models (Random Forest & Multi-Layer Perceptron) to accurately estimate delivery times dynamically.

---

## 🎯 Features

*   **Realistic Data Generation**: Synthesizes highly detailed food delivery logistics including geospatial data, weather, traffic, and rider behavior.
*   **Interactive EDA Dashboard**: Explore distributions, box-plots, and feature correlation directly in your browser.
*   **Machine Learning Models**:
    *   🌲 **Random Forest Regressor** (for robust tabular interpretation)
    *   🧠 **MLP Neural Network** (for complex non-linear pattern matching)
*   **Real-time Predictions**: Input custom delivery scenarios and see how environmental variables affect the final ETA using an ensemble technique.
*   **Model Comparison**: Side-by-side performance analysis with metrics like MAE, RMSE, and $R^2$ Score.

---

## 📸 Screenshots

*(Replace these placeholder links with actual screenshots of your running app)*

<p align="center">
  <b>Landing Page & Dashboard</b><br>
  <code>[Insert Screenshot Image Here]</code>
</p>

<p align="center">
  <b>Real-Time ETA Predictor</b><br>
  <code>[Insert Screenshot Image Here]</code>
</p>

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd eta_prediction_app
   ```

2. **Set up a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage Guide

1. **Start the Application:**
   ```bash
   streamlit run streamlit_app.py
   ```
   *This will open the app locally in your default web browser.*

2. **Generate the Data:**
   Use the left sidebar to configure the `Dataset Size` and preferred test splits. Click **"🚀 Generate Data & Train Models"** to initialize the simulation.

3. **Explore the Tabs:**
   *   **📊 Data Overview:** View the synthesized dataset, statistical summaries, and metrics.
   *   **🔍 EDA:** Check out data visualizations that highlight feature overlaps and target impacts.
   *   **🤖 Model Training:** Build the Random Forest & MLP models over the customized data split.
   *   **📈 Model Performance:** Compare the accuracy and reliability of both models directly.
   *   **🎯 Real-time Prediction:** Tinker with customized delivery orders representing complex conditions to generate instant ETA estimates.

---

## 💼 Business Impact

Correctly predicting Estimated Time of Arrival directly impacts logistics efficiency and customer satisfaction. This system serves as a functional demonstration of applying data science to optimize operational assignment based on fluctuating environmental variables (traffic/weather) and individual historical performances (rider speeds/ratings).

---
*Created as a showcase of modern MLOps, Data Science, and UI practices.*
