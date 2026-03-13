from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor
from datasets import load_dataset

model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")

import librosa

audio, sr = librosa.load("./xy.wav", sr=44100)

processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")

model_output = model.generate(input_features=inputs["input_features"], composer="composer1")

tokenizer_output = processor.batch_decode(
    token_ids=model_output, feature_extractor_output=inputs
)

tokenizer_output = tokenizer_output["pretty_midi_objects"][0]
tokenizer_output.write("./midi_output.mid")
