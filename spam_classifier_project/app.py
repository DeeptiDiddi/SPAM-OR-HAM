from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load saved model and transformers
model = joblib.load('spam_model.pkl')
tfidf = joblib.load('tfidf.pkl')
poly = joblib.load('poly.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data['message']
    tfidf_features = tfidf.transform([message])
    poly_features = poly.transform(tfidf_features.toarray())
    prediction = model.predict(poly_features)[0]
    result = "Spam ❌" if prediction == 1 else "Ham ✅"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
