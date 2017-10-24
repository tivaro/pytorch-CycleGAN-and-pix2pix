import torch.utils.data as data
from PIL import Image
from . import transforms

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_transform(opt):
    transform_list = []

    if opt.audio_length:
        transform_list.append(transforms.FixLength(opt.audio_sizes['n_samples']))
    transform_list.append(transforms.Spectrogram(opt.audio_options))
    transform_list.append(transforms.Complex2Real(opt.phase, opt.amplitude))
    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)