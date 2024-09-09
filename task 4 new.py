import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        # Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the CSV file
file_path = 'C:/Users/DEBPRIYA/OneDrive/Desktop/Task 4/spam.csv'  # Update the path if necessary
data = pd.read_csv(file_path, encoding='latin-1')

# Drop unnecessary columns and rename for convenience
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Encode labels (spam = 1, ham = 0)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Use TF-IDF to vectorize the text data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train and evaluate Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
nb_report = classification_report(y_test, y_pred_nb)
print("Naive Bayes Report:")
print(nb_report)
# Train and evaluate Logistic Regression classifier
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
lr_report = classification_report(y_test, y_pred_lr)
print("Logistic Regression Report:")
print(lr_report)
# Train and evaluate Support Vector Machine classifier
svm_model = SVC(random_state=42)
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)
svm_report = classification_report(y_test, y_pred_svm)
print("SVM Report:")
print(svm_report)
