import pandas as pd
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Download NLTK stopwords
nltk.download('stopwords')

# Load dataset (Ensure this CSV has "text" and "generated" columns)
df = pd.read_csv("Training_Essay_Data.csv")  

# Preprocessing
X = df["text"]
y = df["generated"]  # 1 = AI-generated, 0 = Human-written

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a text processing & classification pipeline
model = make_pipeline(TfidfVectorizer(stop_words='english', max_features=5000), MultinomialNB())

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "ai_text_detector.pkl")
print("Model trained and saved successfully!")
