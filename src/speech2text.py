import pathlib

import torchaudio
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration


model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


def get_transcription(file_name: pathlib.Path) -> str:
    data_waveform, sampling_rate = torchaudio.load(file_name)
    data_waveform = torchaudio.functional.resample(data_waveform, orig_freq=sampling_rate, new_freq=16000)
    inputs = processor(data_waveform[0], sampling_rate=16000, return_tensors="pt")
    generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return " ".join(transcription)
