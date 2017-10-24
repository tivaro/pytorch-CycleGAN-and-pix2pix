###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

import librosa
import os
import os.path

AUDIO_EXTENSIONS = [
    '.wav',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def make_dataset(dir):
    audiofiles = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                audiofiles.append(path)

    return audiofiles


def default_loader(path):
    y, sr = librosa.load(path, mono=True)
    return y

class AudioFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        audiofiles = make_dataset(root)
        if len(audiofiles) == 0:
            raise(RuntimeError("Found 0 audiofiles in: " + root + "\n"
                               "Supported audio extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.audiofiles = audiofiles
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.audiofiles[index]
        audio = self.loader(path)
        if self.transform is not None:
            audio = self.transform(audio)
        if self.return_paths:
            return audio, path
        else:
            return audio

    def __len__(self):
        return len(self.audiofiles)
