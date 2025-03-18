import torch
import torchaudio
from melbanks_orig import LogMelFilterBanks

wav_path = "ezyZip.wav"

signal, sr = torchaudio.load(wav_path)

melspec = torchaudio.transforms.MelSpectrogram(
    hop_length=160,
    n_mels=80
)(signal)

logmelbanks = LogMelFilterBanks()(signal)

assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)
