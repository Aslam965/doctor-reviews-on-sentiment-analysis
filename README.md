# Doctor-Review-Sentiment-Analysis

Doctor Review Sentiment Analysis Project
Project Overview

This project focuses on analyzing sentiments in doctor reviews using machine learning models. The goal is to uncover the hidden sentiments in patient feedback and to understand the overall sentiment towards healthcare providers.
Dataset

The dataset includes doctor reviews, each tagged with a sentiment label (positive or negative). The dataset is structured with textual reviews, sentiment labels (1 for positive, 0 for negative), and tags indicating the review sentiment.

## Methodology
The project follows these key steps:

* Data Cleaning and Preprocessing: The data is cleaned and preprocessed for analysis, which includes converting text to lowercase and removing punctuation.
* Data Visualization: Utilizing word clouds and sentiment distribution charts to explore prevalent themes and sentiment balance in the reviews.
* Machine Learning Models: Implementing Logistic Regression, Naive Bayes, and SVM to conduct sentiment analysis. The performance of each model is evaluated based on accuracy and insights from the confusion matrix.

## Key Findings
* Model Performance: Logistic Regression emerged as the top model with an accuracy of 93.10%, followed by SVM (89.66%) and Naive Bayes (82.76%).
* Review Length Analysis: A correlation was found between the length of the reviews and their sentiments. Negative reviews tend to be longer than positive ones.
* Insights from Visualization: Word clouds and sentiment distribution charts provided a deeper understanding of common themes in reviews and the overall sentiment distribution.

## Application
The project includes a Streamlit application that allows users to input doctor reviews and receive sentiment predictions. This application uses the trained logistic regression model and TF-IDF vectorizer for sentiment analysis.
Repository Contents

* DoctorReviews Analysis.ipynb: Jupyter notebook containing the analysis and model training.
* app.py: Streamlit application script.
* logistic_regression_model.pkl: Trained Logistic Regression model.
* tfidf_vectorizer.pkl: Trained TF-IDF Vectorizer.
* data/: Folder containing the dataset used in the project.

## How to Use
* Clone the repository.
* Install the required dependencies.
* Run app.py to start the Streamlit application.
* Follow the instructions in the app to input a review and get a sentiment prediction.

## Dependencies
* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Matplotlib
* Joblib

## Author
Matthew Horvath

License
This project is licensed under the MIT License.
