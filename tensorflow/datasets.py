import numpy as np
import os
import os.path
import logging
import pickle
import tensorflow as tf
from tensorflow import keras

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def unpickle(file):
    """Unpickle CIFAR-10 batch file"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10_batch(batch_path):
    """Load a single CIFAR-10 batch file"""
    batch_dict = unpickle(batch_path)
    data = batch_dict[b'data']
    labels = np.array(batch_dict[b'labels'])
    
    # Reshape data from (10000, 3072) to (10000, 32, 32, 3)
    data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)
    
    return data, labels


def load_cifar10_from_local(datadir, train=True):
    """Load CIFAR-10 data from local batch files"""
    if train:
        # Load all 5 training batch files (data_batch_1 to data_batch_5)
        # Try multiple possible base paths
        possible_base_paths = [
            datadir,
            os.path.join(datadir, 'cifar-10-batches-py'),
            './data/cifar-10-batches-py',
        ]
        
        base_path = None
        for path in possible_base_paths:
            test_file = os.path.join(path, 'data_batch_1')
            if os.path.exists(test_file):
                base_path = path
                break
        
        if base_path is None:
            raise FileNotFoundError(f"CIFAR-10 batch files not found. Tried base paths: {possible_base_paths}")
        
        # Load all 5 training batches
        all_data = []
        all_labels = []
        for i in range(1, 6):
            batch_file = os.path.join(base_path, f'data_batch_{i}')
            if not os.path.exists(batch_file):
                raise FileNotFoundError(f"CIFAR-10 batch file not found: {batch_file}")
            data, labels = load_cifar10_batch(batch_file)
            all_data.append(data)
            all_labels.append(labels)
        
        # Concatenate all batches
        data = np.concatenate(all_data, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
    else:
        # Load test batch
        possible_paths = [
            os.path.join(datadir, 'test_batch'),
            os.path.join(datadir, 'cifar-10-batches-py', 'test_batch'),
            './data/cifar-10-batches-py/test_batch',
        ]
        batch_path = None
        for path in possible_paths:
            if os.path.exists(path):
                batch_path = path
                break
        if batch_path is None:
            raise FileNotFoundError(f"CIFAR-10 test batch file not found. Tried: {possible_paths}")
        data, labels = load_cifar10_batch(batch_path)
    
    return data, labels


class CIFAR10_truncated:
    """CIFAR-10 dataset class for TensorFlow"""

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=True):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # Load from local batch file
        # If root is a file path, use its directory
        if os.path.isfile(self.root):
            root_dir = os.path.dirname(self.root)
        else:
            root_dir = self.root
        data, target = load_cifar10_from_local(root_dir, self.train)
        
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # Convert to tensor and apply transforms
        img = tf.constant(img, dtype=tf.float32)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            target = tf.constant(target, dtype=tf.int64)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated:
    """CIFAR-100 dataset class for TensorFlow"""

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # For CIFAR-100, you would need to implement similar loading logic
        # For now, return empty arrays
        raise NotImplementedError("CIFAR-100 loading from local files not implemented yet")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = tf.constant(img, dtype=tf.float32)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            target = tf.constant(target, dtype=tf.int64)

        return img, target

    def __len__(self):
        return len(self.data)
