from pathlib import Path
import joblib

root = Path(r"E:\Brandpulse_AI")

model_path = root / "models" / "logistic_regression.pkl"
vectorizer_path = root / "models" / "tfidf_vectorizer.pkl"

print("Model file exists:", model_path.exists())
print("Vectorizer file exists:", vectorizer_path.exists())

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("Model loaded:", type(model).__name__)
print("Vectorizer loaded:", type(vectorizer).__name__)

sample_text = ["I love this product, it is amazing!"]
sample_features = vectorizer.transform(sample_text)
prediction = model.predict(sample_features)

print("Prediction:", prediction[0])