
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load and prepare dataset
df = pd.read_csv("C:/Users/deept/OneDrive/Desktop/spam_classifier_project/spam[1].csv")
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(max_features=300)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_tfidf.toarray())
X_test_poly = poly.transform(X_test_tfidf.toarray())

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_poly, y_train)

# Predict on test data
y_pred = model.predict(X_test_poly)

# Show accuracy and classification report
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and transformers
joblib.dump(model, 'spam_model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(poly, 'poly.pkl')

print("\n✅ Model, TF-IDF, and Polynomial Features saved successfully.")
