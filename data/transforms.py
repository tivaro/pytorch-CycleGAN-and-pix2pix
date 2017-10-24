import numpy as np
import librosa, torch
from torchvision.transforms import Compose


class FixLength(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, audio_tensor):
        return librosa.util.fix_length(audio_tensor, self.length)


class Spectrogram(object):
    """ Represents sampled audio signal as complex spectogram
    """

    def __init__(self, audio_opt):
        if audio_opt['spectrogram'] != 'linear':
            assert audio_opt['n_freq']

        self.audio_opt = audio_opt

    def __call__(self, audio):

        spectrogram_type = self.audio_opt['spectrogram']
        n_bins = self.audio_opt['n_freq']

        if spectrogram_type == 'cqt':
            return librosa.cqt(audio, n_bins=n_bins,
                               n_fft=self.audio_opt['n_fft'],
                               win_length=self.audio_opt['win_length'],
                               hop_length=self.audio_opt['hop_length'],
                               center=self.audio_opt['center'])

        S = librosa.stft(audio, n_fft=self.audio_opt['n_fft'],
                         win_length=self.audio_opt['win_length'],
                         hop_length=self.audio_opt['hop_length'],
                         center=self.audio_opt['center'])

        if spectrogram_type == 'linear':
            return S

        if spectrogram_type == 'mel':
            return librosa.feature.melspectrogram(S=S, n_mels=n_bins)

        raise NotImplementedError('Spectrogram type {} not implemented'.format(self.spectrogram_type))


class Complex2Real(object):
    """ Represents complex spectrogram as real spectogram (magnitude) + phase
    (phase is optional and has different representations)
    """

    def __init__(self, phase_representation, magnitude_scale='log'):
        self.phase_representation = phase_representation
        self.magnitude_scale = magnitude_scale

    def __call__(self, S):

        magnitude, phase = librosa.magphase(S)

        if self.magnitude_scale == 'log':
            librosa.core.amplitude_to_db(magnitude)
        elif self.magnitude_scale == 'linear':
            pass
        else:
            raise NotImplementedError('magnitude_scale {} not implemented'.format(self.magnitude_scale))


        if self.phase_representation == 'angle':
            channels = [magnitude, np.angle(phase)]

        elif self.phase_representation == 'unwrapped_angle':
            channels = [magnitude, np.unwrap(np.angle(phase))]

        elif self.phase_representation == 'none':
            channels = [magnitude]

        elif self.phase_representation == 'cartesian':
            channels = [np.real(S), np.imag(S)]

        else:
            raise NotImplementedError('Phase representation {} not implemented'.format(self.phase_representation))

        return np.stack(channels, axis=0)


class ToTensor(object):
    def __call__(self, spectrograms):
        return torch.from_numpy(spectrograms)