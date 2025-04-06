import torch
import torch.nn as nn
import math

class LogMelFilterBanks(nn.Module):
    def __init__(self, 
                 n_fft: int = 400,
                 hop_length: int = 160,
                 n_mels: int = 80,
                 samplerate: int = 16000,
                 f_min: float = 0.0,
                 f_max: float = None,
                 power: float = 2.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.samplerate = samplerate
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else samplerate / 2
        self.power = power
        self.window = torch.hann_window(n_fft)
        self.mel_fbanks = self._init_melscale_fbanks()

    def _init_melscale_fbanks(self):
        n_freqs = self.n_fft // 2 + 1
        bin_freqs = torch.linspace(0, self.samplerate / 2, n_freqs)
        def hz_to_mel(f): 
            return 2595 * math.log10(1 + f / 700)
        def mel_to_hz(m): 
            return 700 * (10 ** (m / 2595) - 1)
        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_freqs = bin_freqs.unsqueeze(0)
        hz_points = hz_points.unsqueeze(1)
        lower = (bin_freqs - hz_points[:-2]) / (hz_points[1:-1] - hz_points[:-2] + 1e-10)
        upper = (hz_points[2:] - bin_freqs) / (hz_points[2:] - hz_points[1:-1] + 1e-10)
        fb = torch.clamp(torch.min(lower, upper), min=0)
        return fb

    def spectrogram(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        stft = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(x.device),
            center=True,
            pad_mode='reflect',
            return_complex=True
        )
        return stft.abs().pow(self.power)

    def forward(self, x: torch.Tensor):
        power_spec = self.spectrogram(x)
        mel_spec = torch.matmul(self.mel_fbanks.to(power_spec.device), power_spec)
        return torch.log(mel_spec + 1e-6)
