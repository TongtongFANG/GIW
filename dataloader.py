import os
import os.path
import errno
import codecs
import numpy as np
import random
import torch
import torch.utils.data as data
from PIL import Image


def color_grayscale_arr(arr, color):
    """Converts grayscale image to red/green/blue image"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if color == 0:
        arr = np.concatenate([arr,
                              np.zeros((h, w, 2), dtype=dtype)], axis=2)
    elif color == 1:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype), arr,
                              np.zeros((h, w, 1), dtype=dtype)], axis=2)
    elif color == 2:
        arr = np.concatenate([np.zeros((h, w, 2), dtype=dtype), arr], axis=2)

    return arr


# Dataset for colored MNIST
class ColorMNIST(data.Dataset):
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, val=False, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.val = val
        self.dataset = 'mnist'
        random.seed(100)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))

            if self.val:
                idx_all = np.arange(len(self.train_labels))
                idx_select = []
                for i in range(10):
                    idx = (self.train_labels == i)
                    idx_select.extend(random.sample(idx_all[idx].tolist(), 2))

                self.val_data = self.train_data[idx_select]
                self.val_labels = self.train_labels[idx_select]
                self.val_colored_data = self.prepare_rgb_data(self.val_data)


            else:
                self.train_colored_data = []
                for idx, image in enumerate(self.train_data):
                    colored_arr = color_grayscale_arr(np.array(image), color=0)
                    self.train_colored_data.append(colored_arr)

        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            self.test_colored_data = self.prepare_rgb_data(self.test_data)

    def __getitem__(self, index):
        if self.train:
            if self.val:
                img, target = self.val_colored_data[index], self.val_labels[index]
            else:
                img, target = self.train_colored_data[index], self.train_labels[index]
        else:
            img, target = self.test_colored_data[index], self.test_labels[index]

        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            if self.val:
                return len(self.val_colored_data)
            else:
                return len(self.train_colored_data)
        else:
            return len(self.test_colored_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def prepare_rgb_data(self, data):
        num_total = len(data)
        idx_all = np.arange(num_total)
        random.shuffle(idx_all)
        num_group = int(np.round(num_total / 3))
        r_group = idx_all[:num_group]
        g_group = idx_all[(num_group + 1):(2 * num_group + 1)]
        # b_group = idx_all[(2 * num_group + 2):]

        color_data = []

        for idx, image in enumerate(data):
            if idx in r_group:
                colored_arr = color_grayscale_arr(np.array(image), color=0)
            elif idx in g_group:
                colored_arr = color_grayscale_arr(np.array(image), color=1)
            else:
                colored_arr = color_grayscale_arr(np.array(image), color=2)

            color_data.append(colored_arr)

        return color_data


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed.copy()).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed.copy()).view(length, num_rows, num_cols)

