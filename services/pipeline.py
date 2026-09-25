from services.speech_to_text import transcribe_audio
from services.translate import translate_text


def process_audio(audio_path):
    # 1. Speech → Kannada text
    transcript = transcribe_audio(audio_path)

    # 2. Kannada text → English
    translation = translate_text(transcript)

    return {
        "transcript": transcript,
        "translation": translation,
        "source_language": "kn",
        "target_language": "en"
    }


if __name__ == "__main__":
    audio_file = "test_audio/nidsgey.wav"

    result = process_audio(audio_file)

    print("\n--- FINAL RESULT ---")
    print("Kannada:", result["transcript"])
    print("English:", result["translation"])