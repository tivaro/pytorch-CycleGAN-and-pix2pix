import os.path
from data.base_dataset import BaseDataset, get_transform
from data.audio_folder import make_dataset, default_loader
from PIL import Image
import PIL
import random
from . import transforms

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.fold + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.fold + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_audio = default_loader(A_path)
        B_audio = default_loader(B_path)

        A = self.transform(A_audio)
        B = self.transform(B_audio)

        assert A.size(-1)==self.opt.audio_sizes['n_time_bins']
        assert B.size(-1)==self.opt.audio_sizes['n_time_bins']

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
