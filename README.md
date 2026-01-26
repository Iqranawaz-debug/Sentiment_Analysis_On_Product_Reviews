# Sentiment Analysis for Product Reviews

## Project Overview
This project focuses on applying machine learning techniques to interpret and classify emotions expressed in textual product reviews. The primary objective is to categorize sentiments into three main classes: **Positive**, **Negative**, and **Neutral**.

By automating sentiment classification, this system helps businesses monitor brand reputation, evaluate customer satisfaction, and identify trends based on customer feedback.

## Dataset
The project uses the **Amazon Fine Food Reviews dataset** obtained from Kaggle.

- **Source:** A large collection of food-related product reviews  
- **Columns:**  
  - `Text` – the customer review  
  - `Sentiment` – the sentiment label  
- **Preprocessing Steps:**  
  - Removal of stopwords  
  - Elimination of punctuation and special characters  
  - Conversion of text to lowercase  
  - Handling missing values  

## Methodology
The project follows a structured machine learning pipeline:

1. **Data Collection:** Gathering raw textual data  
2. **Data Preprocessing:** Cleaning and normalizing text  
3. **Feature Extraction:** Transforming text into numerical vectors using **TF-IDF**  
4. **Model Training:** Training multiple machine learning models  
5. **Model Evaluation:** Measuring performance using Accuracy, Precision, Recall, and F1-score  

## Models Implemented
The following machine learning models were implemented and compared:

- **Support Vector Machine (SVM):** Identifies an optimal hyperplane for sentiment classification  
- **Random Forest:** An ensemble learning method that improves accuracy and reduces overfitting  
- **Logistic Regression:** A linear classification model suitable for multi-class sentiment analysis  
- **Naive Bayes:** A probabilistic model effective for large-scale text data  

## Results
- **Best Performing Models:** **SVM** and **Random Forest** achieved the highest accuracy after hyperparameter tuning  
- **Moderate Performance:** Logistic Regression  
- **Variable Performance:** Naive Bayes showed fluctuating results on test data  

## Ethical Considerations
- No private or sensitive user data was used  
- Dataset preprocessing aimed to reduce bias and ensure fairness  

