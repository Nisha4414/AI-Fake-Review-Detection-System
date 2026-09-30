from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load AI model and TF-IDF vectorizer
model = joblib.load("calibrated_svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return send_file("app.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    review = data.get("review", "")

    if not review.strip():
        return jsonify({
            "error": "Please enter a review."
        })

    review_tfidf = tfidf.transform([review])

    prediction = model.predict(review_tfidf)[0]
    probabilities = model.predict_proba(review_tfidf)[0]

    if prediction == 1:
        result = "Potentially Fake Review"
        confidence = probabilities[1] * 100
    else:
        result = "Likely Original Review"
        confidence = probabilities[0] * 100

    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
