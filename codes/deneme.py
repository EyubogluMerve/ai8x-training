import numpy as np
import librosa
import pytsmod as tsm
import soundfile as sf
import os



record_pth = "/home/merveeyuboglu/Github/ai8x-training/data/KWS/raw/backward/0a2b400e_nohash_0.wav"
record, fs = librosa.load(record_pth, offset=0, sr=None)

print(record.shape)

record = np.pad(record, [0, 16384 - record.size])
print(record.shape)
record = record.reshape((len(record),1))
print(record.shape)