# Spam Detection API

A production-ready Spam Detection system built using Machine Learning and FastAPI.

## 🚀 Tech Stack
- Python
- Scikit-learn
- FastAPI
- Docker

## 📌 Features
- Trained Multinomial Naive Bayes model
- Text vectorization using CountVectorizer
- REST API endpoint for real-time prediction
- Dockerized for deployment

## 🧠 Model Details
- Algorithm: Multinomial Naive Bayes
- Accuracy: ~98.5%
- Dataset: SMS Spam Collection Dataset

## 🔌 API Endpoint

POST `/predict`

Example Request:
```json
{
  "text": "Free money offer now"
}