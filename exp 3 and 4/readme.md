# Spam Classification Project

This project implements and evaluates **spam vs ham classification** using both **Logistic Regression** and **Multinomial Naive Bayes**, including models built from scratch and scikit-learn implementations. Various vectorization techniques are compared, and results are analyzed using standard evaluation metrics.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Implemented Models](#implemented-models)
- [Experiments](#experiments)
- [Results](#results)

---

## Overview
The goal of this project is to classify SMS messages into **spam** and **ham** categories. The project explores:

- Implementing **Logistic Regression** and **Multinomial Naive Bayes** from scratch.
- Using scikit-learn versions of the models for comparison.
- Testing **CountVectorizer** and **TF-IDF Vectorizer** as feature representations.
- Evaluating performance with Accuracy, Precision, Recall, F1-score, and Confusion Matrices.
- Analyzing the effect of **regularization (λ)** on Logistic Regression.

---

## Dataset
- The dataset consists of SMS messages labeled as **ham** (0) or **spam** (1).
- Preprocessing steps include:
  - Text cleaning and stemming.
  - Mapping labels: `'ham' → 0`, `'spam' → 1`.
- Feature matrices are created using:
  - **CountVectorizer**
  - **TF-IDF Vectorizer**

---

## Implemented Models

### Logistic Regression
- **From scratch:** Gradient descent optimization with optional L2 regularization.
- **Scikit-learn:** `LogisticRegression` with regularization support.

### Multinomial Naive Bayes
- **From scratch:** Laplace smoothing (α) applied for probability estimation.
- **Scikit-learn:** `MultinomialNB` for comparison.

---

## Experiments
1. Train **Logistic Regression** with:
   - Vectorizers: CountVectorizer, TF-IDF
   - λ values: 0, 0.01, 0.1, 1
   - Optional feature scaling
2. Train **Multinomial Naive Bayes** with:
   - Vectorizers: CountVectorizer, TF-IDF
   - Laplace smoothing α = 1
3. Evaluate models using:
   - **Accuracy**, **Precision**, **Recall**, **F1-score**
   - **Confusion Matrix**
4. Compare all results in a **summary table** for easy reference.

---

## Results| Source             | Model                         | Vectorizer   | λ    | Accuracy | Precision | Recall  | F1     |
|-------------------|-------------------------------|-------------|------|----------|-----------|---------|--------|
| Logistic Scratch   | Logistic Regression           | Count       | 0.0  | 0.9777   | 0.9134    | 0.9062  | 0.9098 |
| Logistic Scratch   | Logistic Regression           | Count       | 0.01 | 0.9806   | 0.9286    | 0.9141  | 0.9213 |
| Logistic Scratch   | Logistic Regression           | Count       | 0.1  | 0.9777   | 0.9134    | 0.9062  | 0.9098 |
| Logistic Scratch   | Logistic Regression           | Count       | 1.0  | 0.9777   | 0.9134    | 0.9062  | 0.9098 |
| Logistic Scratch   | Logistic Regression           | TF-IDF      | 0.0  | 0.9671   | 0.8264    | 0.9297  | 0.8750 |
| Logistic Scratch   | Logistic Regression           | TF-IDF      | 0.01 | 0.9661   | 0.8345    | 0.9062  | 0.8689 |
| Logistic Scratch   | Logistic Regression           | TF-IDF      | 0.1  | 0.9641   | 0.8182    | 0.9141  | 0.8635 |
| Logistic Scratch   | Logistic Regression           | TF-IDF      | 1.0  | 0.9632   | 0.8082    | 0.9219  | 0.8613 |
| Logistic sklearn   | Logistic Regression (sklearn) | Count       | 0.0  | 0.9729   | 0.9808    | 0.7969  | 0.8793 |
| Logistic sklearn   | Logistic Regression (sklearn) | Count       | 0.01 | 0.9767   | 0.9127    | 0.8984  | 0.9055 |
| Logistic sklearn   | Logistic Regression (sklearn) | Count       | 0.1  | 0.9767   | 0.9262    | 0.8828  | 0.9040 |
| Logistic sklearn   | Logistic Regression (sklearn) | Count       | 1.0  | 0.9729   | 0.9808    | 0.7969  | 0.8793 |
| Logistic sklearn   | Logistic Regression (sklearn) | TF-IDF      | 0.0  | 0.9661   | 0.9604    | 0.7578  | 0.8472 |
| Logistic sklearn   | Logistic Regression (sklearn) | TF-IDF      | 0.01 | 0.9641   | 0.8321    | 0.8906  | 0.8604 |
| Logistic sklearn   | Logistic Regression (sklearn) | TF-IDF      | 0.1  | 0.9651   | 0.8433    | 0.8828  | 0.8626 |
| Logistic sklearn   | Logistic Regression (sklearn) | TF-IDF      | 1.0  | 0.9661   | 0.9604    | 0.7578  | 0.8472 |
| Naive Bayes Scratch| Naive Bayes                   | CountVector | ---  | 0.9612   | 0.8099    | 0.8984  | 0.8519 |
| Naive Bayes Scratch| Naive Bayes                   | TfidfVector | ---  | 0.9603   | 0.9888    | 0.6875  | 0.8111 |
