import os

os.add_dll_directory(
    r"C:\Users\harsh\Downloads\ffmpeg-9.0.2-full_build-shared\bin"
)
import torch
import torchaudio
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


def transcribe_audio(audio_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model = model.to(device)
    model.eval()

    waveform, sample_rate = torchaudio.load(audio_path)

    # Make sure audio is 16 kHz
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            16000
        )

    # Convert stereo → mono if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.to(device)

    with torch.no_grad():
        output = model(
            waveform,
            lang="kn",
            decoding="ctc"
        )

    return output


if __name__ == "__main__":
    audio_file = "test_audio/nidsgey.wav"

    transcript = transcribe_audio(audio_file)

    print("\nKannada Transcript:")
    print(transcript)
    print()