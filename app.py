from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
from data import allData

app = Flask(__name__)

# === Load model ===
model = load_model("ewaste_model.keras")

# === Define categories in correct order ===
categories = [
    'camera', 'keyboard', 'laptop', 'microwave',
    'mobile', 'mouse', 'smartwatch', 'tv'
]

@app.route("/")
def home():
    return jsonify({"message": "Aitana ML API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    try:
        # preprocess image
        img = Image.open(BytesIO(file.read())).convert("RGB").resize((224, 224))
        img_arr = image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        # predict
        pred = model.predict(img_arr)[0]
        idx = np.argmax(pred)
        label = categories[idx]
        confidence = float(pred[idx])

        # optional log for debugging
        print(f"[DEBUG] Prediction vector: {pred}")
        print(f"[DEBUG] Predicted: {label} ({confidence:.2%})")

        return jsonify({
            "label": label,
            "confidence": f"{confidence:.2%}",
            "funFacts": allData[label]["funFacts"],
            "reuseIdeas": allData[label]["ideas"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
