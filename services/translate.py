import sys
import torch
from IndicTransToolkit import IndicProcessor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

ip = IndicProcessor(inference=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL,
    trust_remote_code=True
).to(device)


def translate_text(kannada_text):
    text = [kannada_text]

    batch = ip.preprocess_batch(
        text,
        src_lang="kan_Knda",
        tgt_lang="eng_Latn"
    )

    inputs = tokenizer(
        batch,
        padding="longest",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=256,
            num_beams=1,
            use_cache=False
        )

    decoded = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True
    )

    translation = ip.postprocess_batch(
        decoded,
        lang="eng_Latn"
    )

    return translation[0]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python services\\translate.py \"Kannada text\"")
        sys.exit(1)

    kannada_text = sys.argv[1]
    english_text = translate_text(kannada_text)

    print("Kannada:", kannada_text)
    print("English:", english_text)