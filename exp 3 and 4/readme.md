# Spam Classification

This project demonstrates spam detection using machine learning techniques in Python. The workflow and results are documented in the `spam classification.ipynb` notebook.

## Overview

- **Objective:** Classify messages as spam or not spam.
- **Dataset:** SMS Spam Collection Dataset.
- **Techniques:** Data preprocessing, feature extraction (TF-IDF), model training, and evaluation.

## Steps

1. **Data Loading:** Import and inspect the dataset.
2. **Preprocessing:** Clean text, remove stopwords, and tokenize.
3. **Feature Extraction:** Convert text to numerical features using TF-IDF.
4. **Model Training:** Train classifiers (e.g., Naive Bayes, Logistic Regression).
5. **Evaluation:** Assess model performance using accuracy, precision, recall, and F1-score.

## Results

- Achieved high accuracy in spam detection.
- Naive Bayes performed well for this text classification task.

## Usage

1. Clone the repository.
2. Open `spam classification.ipynb` in Jupyter Notebook.
3. Run all cells to reproduce the results.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- nltk
- jupyter

## References

- [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
