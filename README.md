# youtube-adview-predictor


# **YouTube AdView Prediction** 

Predicting the **number of ad views** on YouTube videos using **machine learning**.
This project focuses on building a regression model trained on video metadata to estimate the number of **ad impressions** a video can potentially generate.

---

## ** Project Overview**

YouTube videos earn revenue through ads, and predicting the number of ad views can help creators and marketers estimate potential earnings.
This project uses **metadata-driven regression** to predict ad view counts.

**Key Objectives:**

* Perform **EDA** (Exploratory Data Analysis) on the dataset
* Preprocess and clean data for model training
* Train a **regression model** for predicting YouTube ad views
* Build a simple **Flask-based web app** to make predictions interactively
* Experiment with multiple **deployment strategies**

---

## ** Dataset**

The dataset contains YouTube video metadata, including:

* **Video ID**
* **Title** & **Category**
* **Duration**
* **View count**
* **Likes, dislikes, comments**
* **Ad view counts** *(target variable)*

**Files:**

* `train.csv` → Training dataset
* `test.csv` → Testing dataset

---

## ** Tech Stack**

* **Language:** Python 
* **Libraries:**

  * `scikit-learn` → Model training
  * `numpy` & `pandas` → Data handling
  * `matplotlib` & `seaborn` → Visualization
  * `flask` → Web app
  * `joblib` → Model serialization
* **Deployment Experiments:**

  * Hugging Face Spaces *(final target)*
  * Replicate API *(explored but not finalized)*
  * Docker *(experimented but not pursuing further)*

---

## ** The Journey**

This project is **not just a model** — it’s been a **complete learning experience**.
We went through **multiple iterations** of building, breaking, fixing, and re-building:

* **First Version** → Basic model training + prediction pipeline 
* **Second Version** → Cleaner code, better structure, started preparing for deployment 
* **Deployment Experiments**:

  * **Replicate**: Tried serving the model via API but faced limitations
  * **Docker & Codespaces**: Experimented but decided not to continue
  * **CMD, PowerShell & Anaconda Prompt**: Fought environment mismatches & dependency issues 
  * **Hugging Face Spaces**: Finally decided to deploy here — currently in progress
* Learned a **lot** about **environments, dependency conflicts, and deployment pitfalls**.

This repo reflects not just the **solution**, but also the **process** behind building it.

---

## ** Deployment**

We experimented with different deployment platforms:

* **Hugging Face Spaces** → *\[Coming Soon]*
* **Replicate API** → *\[Coming Soon]*

Stay tuned for working demos 

---

## ** Future Work**

*  Finalize deployment on **Hugging Face Spaces**
*  Set up a **Replicate API** endpoint
* Improve **Flask UI** for better user experience
* Optimize the model for faster inference

---

## Takeaways**

* Built a complete **ML pipeline** — from training to predictions
* Explored multiple deployment paths before settling on **Hugging Face**
* Learned a lot about **environment handling** and **dependency management**
* Failed, fixed, retried, and learned more every step 

> *“Still experimenting. Still building. Still learning.”* 
