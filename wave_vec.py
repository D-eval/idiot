from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa
import soundfile as sf


source_path = "/Users/broyou/Desktop/笔记本/preprocess/segments/Akibare_0000.wav"
samplerate = 16000

audio, current_sample_rate = librosa.load(source_path, sr=samplerate)



processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

inputs = processor(audio, sampling_rate=16000, return_tensors="pt") # 归一化
outputs = model(**inputs)

features = outputs.extract_features

hidden_state = outputs.last_hidden_state
