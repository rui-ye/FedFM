import paddle.io as data
from PIL import Image
import numpy as np
import paddle.vision
from paddle.vision.datasets import  Cifar10, Cifar100
import pickle
import os
import os.path
import logging
import sys
import copy
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def load_cifar10_batch(file):
    """读取 CIFAR-10 单个 batch（二进制）"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')

    # 数据 shape: (10000, 3072)
    data = dict['data']
    labels = dict['labels']

    # reshape -> (N, 3, 32, 32) -> 再转 (N, 32, 32, 3)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = data.astype(np.uint8)

    return data, np.array(labels, dtype=np.int64)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_cifar_from_binary(root, train=True):
    base = os.path.join(root, 'cifar-10-batches-py')

    def load_file(f):
        with open(os.path.join(base, f), 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
        return dic[b'data'], dic[b'labels']

    data_list, label_list = [], []

    if train:
        for i in range(1, 6):
            data, labels = load_file(f"data_batch_{i}")
            data_list.append(data)
            label_list.extend(labels)
    else:
        data, labels = load_file("test_batch")
        data_list.append(data)
        label_list.extend(labels)

    data = np.vstack(data_list).reshape(-1, 3, 32, 32)
    data = np.transpose(data, (0, 2, 3, 1))  # CHW → HWC
    data = data.astype(np.uint8)

    target = np.array(label_list, dtype=np.uint8)

    return data, target

class Cifar10_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root                      # ./data/cifar-10-python.tar.gz (ignored)
        self.base_folder = "./data/cifar-10-batches-py"
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        """直接读取 CIFAR10 二进制文件，比 Paddle 内置类快很多"""

        if self.train:
            batch_list = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5"
            ]
        else:
            batch_list = ["test_batch"]

        # ------------ 加载全部 batch -------------
        datas = []
        labels = []

        for batch_name in batch_list:
            file_path = os.path.join(self.base_folder, batch_name)
            data, label = load_cifar10_batch(file_path)
            datas.append(data)
            labels.append(label)

        data = np.concatenate(datas, axis=0)
        target = np.concatenate(labels, axis=0)
        print(data[0])
        print(target[0])


        # ------------ 过滤 dataidxs（如果有） -------------
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        """将 index 指定样本的 G/B 通道置 0"""
        self.data[index, :, :, 1] = 0
        self.data[index, :, :, 2] = 0

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)   # Paddle transform 支持 numpy(H,W,C)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

# class Cifar10_truncated(data.Dataset):

#     def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=True):

#         self.root = root
#         self.dataidxs = dataidxs
#         if train:
#             self.train = "train"
#         else:
#             self.train = "test"
#         self.transform = transform
#         self.target_transform = target_transform
#         self.download = download

#         self.data, self.target = self.__build_truncated_dataset__()

#     def __build_truncated_dataset__(self):
#         Cifar_dataobj = Cifar10(self.root, self.train, self.transform, self.download)


#         data_tmp=[]
#         target_tmp = []
#         if self.train == 'train':
#             data_number = 50000
#         else:
#             data_number = 10000
#         for i in range(data_number):
#             img,label = Cifar_dataobj[i]
#             data_tmp.append(img*255)
#             target_tmp.append(label)
#         # print(data_tmp[0])
#         # print(target_tmp[0])
#         data = np.array(data_tmp,dtype=np.uint8).reshape(-1,32,32,3)
#         target = np.array(target_tmp,dtype=np.uint8)
#         # print(data[0])
#         # print(target[0])
#         if self.dataidxs is not None:
#             data = data[self.dataidxs]
#             target = target[self.dataidxs]
#         return data, target



#     def truncate_channel(self, index):
#         for i in range(index.shape[0]):
#             gs_index = index[i]
#             self.data[gs_index, :, :, 1] = 0.0
#             self.data[gs_index, :, :, 2] = 0.0

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.data[index], self.target[index]
#         # img = Image.fromarray(img)
#         # print("Cifar10 img:", img)
#         # print("Cifar10 target:", target)

#         if self.transform is not None:
#             img = self.transform(img)



#         return img, target

#     def __len__(self):
#         return len(self.data)


class Cifar100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        Cifar_dataobj = Cifar100(self.root, self.train, self.transform, self.download)

        if paddle.vision.__version__ == '0.2.1':
            if self.train:
                data, target = Cifar_dataobj.train_data, np.array(Cifar_dataobj.train_labels)
            else:
                data, target = Cifar_dataobj.test_data, np.array(Cifar_dataobj.test_labels)
        else:
            data = Cifar_dataobj.data
            target = np.array(Cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        # print("Cifar10 img:", img)
        # print("Cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)


        return img, target

    def __len__(self):
        return len(self.data)


