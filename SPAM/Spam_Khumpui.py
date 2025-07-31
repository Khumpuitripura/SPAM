# Spam Email Classifier using Naive Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("Downloads/spam_data/spam.csv", encoding='latin-1')

# Keep only the necessary columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels: ham = 0, spam = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data into features and target
X = df['message']
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Convert text data into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict on test data
y_pred = model.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Function to test the model with a custom email
def predict_spam(message):
    msg_vec = vectorizer.transform([message])
    pred = model.predict(msg_vec)[0]
    return "Spam" if pred else "Not Spam"

# Test the function
sample_email = "Congratulations! You've won a free iPhone. Click here to claim now."
print("\nSample Test Message:")
print(f"'{sample_email}' → {predict_spam(sample_email)}")
