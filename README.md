# SPAM
This project is a simple but effective spam email classifier built using Python and scikit-learn. It uses a labeled dataset of emails (spam and ham) and applies natural language processing techniques along with a machine learning algorithm (Multinomial Naive Bayes) to train a model that can predict whether a given email is spam or not.  
PROJECT TITLE: Spam Email Classifier

DESCRIPTION:
This project is a machine learning-based spam email classifier built using Python. It uses a labeled dataset of spam and ham emails to train a model that can predict whether a given message is spam or not.

TECHNOLOGIES USED:
- Python
- Pandas
- scikit-learn
- Streamlit (for UI)

DATASET USED:
- File: spam.csv
- Source: Kaggle
- Location: ./spam_data/spam.csv

FEATURES:
- Text preprocessing (lowercase, remove punctuation, etc.)
- TF-IDF Vectorization
- Naive Bayes classification
- User-friendly Streamlit web app for live predictions

HOW TO RUN:
1. Install required libraries using: pip install -r requirements.txt
2. Run the app with: streamlit run spam_classifier.py
3. Enter any message in the input field to check if it's spam or not.

PROJECT STRUCTURE:
- spam_classifier.py → Main app
- spam_data/spam.csv → Dataset
- requirements.txt → Dependencies

AUTHOR: Khumpui Tripura
YEAR: 2025
