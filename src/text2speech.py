import pathlib
import uuid

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from datasets import load_dataset

from src.config import config

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
voice_coder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")


def get_speech_audio_path(text: str) -> pathlib.Path:
    inputs = processor(text=text, return_tensors="pt")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=voice_coder)
    path = pathlib.Path(config.audio_file_path) / f"{uuid.uuid4()}.ogg"
    sf.write(path, speech.numpy(), samplerate=16000)

    return path
