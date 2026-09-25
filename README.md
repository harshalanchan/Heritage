# Heritage
# Heritage AI4Bharat Service

AI-powered Kannada speech transcription and Kannada-to-English translation service.

## Pipeline

Audio
↓
AI4Bharat Indic-Conformer 600M
↓
Kannada Speech-to-Text
↓
AI4Bharat IndicTrans2 200M
↓
English Translation
↓
JSON Response

## Requirements

- Python 3.13
- NVIDIA GPU recommended
- CUDA-enabled PyTorch
- FFmpeg shared libraries

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd Heritage