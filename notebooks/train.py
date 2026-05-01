import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("data/spam.tsv", sep="\t", header=None)
df.columns = ["label", "text"]

# Convert labels to numeric
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)

print("\nModel trained successfully!")
print("Model Accuracy:", accuracy)
import joblib

# Save model
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model and vectorizer saved successfully!")