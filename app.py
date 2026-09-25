from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from services.pipeline import process_audio
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "temp_audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/api/process-audio", methods=["POST"])
def process_audio_api():

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio = request.files["audio"]

    if audio.filename == "":
        return jsonify({"error": "No file selected"}), 400

    original_name = secure_filename(audio.filename)
    file_path = os.path.join(
        UPLOAD_FOLDER,
        f"{uuid.uuid4()}_{original_name}"
    )

    audio.save(file_path)

    try:
        result = process_audio(file_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )