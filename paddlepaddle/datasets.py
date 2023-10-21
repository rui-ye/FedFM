import paddle.io as data
from PIL import Image
import numpy as np
import paddle.vision
from paddle.vision.datasets import Cifar10

import os
import os.path
import logging
import sys

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass



class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train='train', transform=None, target_transform=None, download=True):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_dataobj = Cifar10(data_file=self.root, mode=self.train, transform=self.transform, download=self.download)
        # if torchvision.__version__ == '0.2.1':
        #     if self.train:
        #         data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
        #     else:
        #         data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        # else:
        #     data = cifar_dataobj.data
        #     target = np.array(cifar_dataobj.targets)
        # data = cifar_dataobj.data
        # target = np.array(cifar_dataobj.targets)

        # if self.dataidxs is not None:
        #     data = data[self.dataidxs]
        #     target = target[self.dataidxs]
        data_tmp=[]
        target_tmp = []
        if self.train == 'train':
            data_number = 50000
        else:
            data_number = 10000
        for i in range(data_number):
            img,label = cifar_dataobj[i]
            data_tmp.append(img)
            target_tmp.append(label)
        data = np.array(data_tmp,dtype=np.uint8).reshape(-1,32,32,3)
        target = np.array(target_tmp,dtype=np.uint8)
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
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)



