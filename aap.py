from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import math

app = Flask(__name__)
CORS(app)

# Load improved AI model and TF-IDF vectorizer
model = joblib.load("improved_svm_model.pkl")
tfidf = joblib.load("improved_tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return send_file("app.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")

    if not review.strip():
        return jsonify({"error": "Please enter a review."})

    review_tfidf = tfidf.transform([review])

    prediction = model.predict(review_tfidf)[0]
    score = model.decision_function(review_tfidf)[0]

    # Approximate confidence based on SVM decision score
    confidence = 1 / (1 + math.exp(-abs(score)))
    confidence = confidence * 100

    if prediction == 1:
        result = "FAKE REVIEW"
    else:
        result = "REAL REVIEW"

    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
