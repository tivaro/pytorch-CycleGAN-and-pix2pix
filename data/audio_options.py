def get_audio_opt(opt_dict):
    audio_opt = dict()

    # Spectrogram opt (librosa defaults)
    audio_opt['n_fft'] = 2048
    audio_opt['win_length'] = audio_opt['n_fft']
    audio_opt['hop_length'] = audio_opt['win_length'] // 4
    audio_opt['center'] = True

    audio_opt['sr'] = 22050

    # Options that should be defined
    audio_opt['spectrogram'] = None
    audio_opt['phase'] = None
    audio_opt['amplitude'] = None
    audio_opt['n_freq'] = None
    audio_opt['audio_length'] = None


    audio_opt.update((k, v) for k, v in opt_dict.items() if k in audio_opt)

    return audio_opt

def get_audio_sizes(audio_opt, audio_duration=None):
    audio_duration = audio_duration or audio_opt['audio_length']

    sizes = dict()

    # Number of frequency bins depends on spectrogram type
    if audio_opt['spectrogram'] == 'linear':
        sizes['n_freq_bins'] = 1 + (audio_opt['n_fft'] // 2)
    else:
        sizes['n_freq_bins'] = audio_opt['n_freq']

    # Calculate the true n_camples but add half a frame lenght for easier audio reconstruction from spectrogram
    sizes['n_samples_unpadded'] = audio_opt['sr'] * audio_duration // 1000
    sizes['n_samples'] = sizes['n_samples_unpadded'] + (audio_opt['n_fft'] // 2)

    # Calculate number of frames based on librosa.core.stft
    n_samples = sizes['n_samples'] + (bool(audio_opt['center']) * audio_opt['n_fft'])
    sizes['n_time_bins'] = 1 + ((n_samples - audio_opt['n_fft']) // audio_opt['hop_length'])

    sizes['n_channels'] = 1 if audio_opt['phase'] == 'none' else 2

    return sizes