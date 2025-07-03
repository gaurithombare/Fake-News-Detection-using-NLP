# 📰 Fake News Detection using NLP and Machine Learning

This project is an end-to-end implementation of a Fake News Detection system using Natural Language Processing (NLP) and supervised machine learning models. The goal is to classify news articles as **Real** or **Fake** based on their textual content.

## 📌 Project Overview

Fake news has become a widespread problem in the digital age, spreading misinformation and creating confusion. In this project, I developed a machine learning model that detects fake news by analyzing text using NLP techniques.

## 🧠 Models Used

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest Classifier

The model with the best accuracy was selected for deployment.

## 🧰 Technologies & Libraries

- Python
- Pandas, NumPy
- NLTK (Natural Language Toolkit)
- Scikit-learn
- Streamlit
- Joblib

## 🗂️ Dataset

The dataset used is a combination of:
- `True.csv` – Real news articles
- `Fake.csv` – Fake news articles  
Both datasets were preprocessed, merged, and balanced manually using additional real news scraped from **NDTV**, **Times of India**, and **Indian Express**.

> 📁 Final dataset: `final_balanced_dataset.csv`

## 🔍 NLP Techniques Used

- Text Cleaning (removal of punctuation, URLs, etc.)
- Stopwords Removal
- Stemming (Porter Stemmer)
- TF-IDF Vectorization

## 📈 Model Evaluation

Each model was evaluated using accuracy and confusion matrix. The best-performing model was saved using Joblib and deployed in the app.

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 96.5%    |
| Naive Bayes        | 95.4%    |
| Random Forest      | 93.7%    |

## 🚀 Project Structure

