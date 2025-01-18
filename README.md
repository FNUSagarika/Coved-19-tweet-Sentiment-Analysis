# **COVID-19 Tweet Sentiment Analysis**  
_Using Machine Learning and Deep Learning Techniques_  

## **Overview**  
This project focuses on analyzing public sentiment during the COVID-19 pandemic by classifying tweets into five sentiment categories:  
- **Extremely Negative** : Severe dissatisfaction, fear, or frustration related to health crises, policies, or economic struggles 
- **Negative** : Moderate concerns or criticism
- **Neutral** : Objective or informational content
- **Positive** :Expressions of approval or satisfaction  
- **Extremely Positive** :Strong optimism or joy

The analysis leverages machine learning and deep learning models to understand public emotions, which can guide decision-making during crises like pandemics.

---

## **Table of Contents**  
- [Problem Statement](#problem-statement)  
- [Data Description](#data-description)  
- [Tools and Technologies Used](#tools-and-technologies-used)  
- [Methodology](#methodology)  
- [Results](#results)  
- [Key Findings](#key-findings)  
- [Challenges and Solutions](#challenges-and-solutions)  
- [How to Use This Repository](#how-to-use-this-repository)  
- [Future Work](#future-work)  
- [Acknowledgments](#acknowledgments)  

---

## **Problem Statement**  
The COVID-19 pandemic led to a surge in discussions on social media, particularly Twitter. These tweets reflect diverse emotions and concerns, making them a valuable source for analyzing public sentiment.  

This project addresses the following question:  
**"How can we effectively classify COVID-19-related tweets to understand public perceptions and emotions?"**

---

## **Data Description**  
- **Datasets Source:** [Training & Testing Dataset](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/data)  
- **Training Dataset:** `Corona_NLP_train.csv` (Contains over 41,000 labeled tweets categorized into predefined sentiment classes. Purpose: To train machine learning models to identify and analyze sentiment patterns effectively)
- **Testing Dataset:** `Corona_NLP_test.csv` (Includes more than 3,900 labeled tweets with predefined sentiment classes. Purpose: To evaluate the accuracy and performance of the trained models on unseen data)  
- **Evaluation Dataset:** [COVID-19_tweets](https://www.kaggle.com/datasets/gpreda/covid19-tweets) (Dataset containing over 179,000 tweets. Purpose: This dataset is utilized to evaluate the generalizability and robustness of the trained models when applied to unseen, real-world data)  

---

## **Tools and Technologies Used**  
- **Programming Language:** Python  
- **Libraries:**  
  - Data Manipulation: `Pandas`, `NumPy`  
  - Visualization: `matplotlib`, `Seaborn`  
  - Text Processing: `NLTK`, `re`, `TensorFlow Tokenizer`  
  - Machine Learning: `Scikit-learn`, `TensorFlow`  
- **Environment:** Google Colab  

---

## **Methodology**  

### **1. Data Preprocessing**  
- Removed irrelevant columns and unnecessary text elements (mentions, hashtags, URLs).  
- Standardized text (converted to lowercase, removed stopwords).  
- Tokenized tweets and encoded sentiments numerically.  

### **2. Exploratory Data Analysis (EDA)**  
- Visualized sentiment distributions, top hashtags, and word clouds.  
- Identified geographical and linguistic biases in the data.  

### **3. Modeling**  
Implemented and compared five models:  
1. Logistic Regression  
2. Multinomial Naive Bayes  
3. Linear Support Vector Classifier (SVC)  
4. Random Forest  
5. Recurrent Neural Network (RNN) with LSTM  

### **4. Evaluation Metrics**  
- Accuracy  
- Precision  
- Recall  
- F1-Score  

---

## **Results**  
- **Top-performing Model:** RNN with LSTM  
  - **Accuracy:** 75.78%  
  - **F1-Score:** 75.97%  

- Traditional models (Logistic Regression, SVM) achieved ~57% accuracy.  
- Random Forest and Naive Bayes struggled with this task due to text data's high dimensionality.  

---

## **Key Findings**  
- Deep learning models are significantly more effective for sentiment analysis tasks due to their ability to capture context and sequential dependencies.  
- Traditional models can be used for lightweight implementations but are less effective for nuanced text data.  

---

## **Challenges and Solutions**  
1. **Challenge:** Imbalanced sentiment distribution.  
   **Solution:** Applied oversampling techniques for balanced training.  

2. **Challenge:** Handling missing data in the "Location" column.  
   **Solution:** Excluded the column due to limited relevance and high missing rates.  
---

## **Acknowledgments**  
- Kaggle for providing the datasets.  
- Libraries and tools that made this analysis possible.  
