import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# Load the dataset
data = pd.read_csv('cleaned_data_combined_modified.csv')

# Fill missing values with a fallback text
data = data.fillna('fallback_text')

# Convert non-string columns to strings and handle empty strings
for column in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']:
    data[column] = data[column].astype(str).apply(lambda x: "fallback_text" if x.strip() == "" else x)

# Adjust token_pattern to include single-character tokens
vectorizers = {}
X_list = []

for column in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']:
    # Use token_pattern to include single-character tokens
    vectorizer = CountVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b')

    try:
        X_col = vectorizer.fit_transform(data[column])

        # Check for empty vocabulary
        if not vectorizer.vocabulary_:
            print(f"Warning: Column {column} resulted in empty vocabulary. Retrying without stop words.")
            vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')  # Retry without stop words
            X_col = vectorizer.fit_transform(data[column])

        vectorizers[column] = vectorizer
        X_list.append(X_col)

    except ValueError as e:
        print(f"Skipping column {column} due to error: {e}")
        continue

# Combine features
if X_list:
    X = hstack(X_list)
else:
    raise ValueError("No valid features found.")

y = data['Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
# Train and evaluate
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')