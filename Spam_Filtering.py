import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the real SMS Spam Collection dataset
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("===== EMAIL & SPAM DETECTOR =====")
print(f"Total messages: {len(df)}")

# Check the dataset
print("\nDataset distribution:")
print(df["label"].value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Create machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("classifier", MultinomialNB())
])

# Train model
print("\nTraining the model...")
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, predictions))

# Confusion matrix
print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, predictions))

# Test your own messages
print("\n===== TEST YOUR MESSAGE =====")
print("Type 'exit' to close the program.")

while True:

    message = input("\nEnter a message: ")

    if message.lower() == "exit":
        print("\nProgram closed.")
        break

    result = model.predict([message])[0]

    if result == "spam":
        print("⚠️ Result: SPAM")
    else:
        print("✅ Result: NOT SPAM")