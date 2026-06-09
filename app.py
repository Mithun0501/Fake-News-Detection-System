from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    confidence = ""

    if request.method == "POST":

        news_text = request.form["news"]

        text_vector = vectorizer.transform([news_text])

        result = model.predict(text_vector)[0]

        print("Prediction Result:", result)

        probabilities = model.predict_proba(text_vector)[0]

        confidence = round(max(probabilities) * 100, 2)

        if result == 1:
            prediction = "🟢 Real News"
        else:
            prediction = "🔴 Fake News"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)