import mindspore.dataset as data
from PIL import Image
import numpy as np
import mindspore
import mindspore.dataset.vision as vision
from mindspore.dataset import  Cifar100Dataset,Cifar10Dataset
import mindspore.dataset.transforms as transforms
from mindspore import dtype as mstype
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



class CIFAR10_truncated():

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        super(CIFAR10_truncated).__init__()
        self.root = root
        self.dataidxs = dataidxs
        if train == True:
            self.train = "train"
        else:
            self.train = "test"
        # self.transform = [transforms.TypeCast(mstype.int32)]
        # self.transform += [transform]
        # self.transform = [vision.Rescale(1.0 / 255.0, 0.0),
        #                   vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        #                                                                     vision.HWC2CHW()]
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target= self.__build_truncated_dataset__()
        

    def __build_truncated_dataset__(self):
        cifar_dataobj = Cifar10Dataset(self.root, usage=self.train)
        # if torchvision.__version__ == '0.2.1':
        #     if self.train:
        #         data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
        #     else:
        #         data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        # else:
        print('start loading cifar10')
        if self.transform!=None:
        #     cifar_dataobj = cifar_dataobj.map(operations=self.transform, input_columns = 'image')
        
            # print(data.get_dataset_size)
            # cifar_dataobj = cifar_dataobj.map(operations=self.transform, input_columns = 'label')
            cifar_dataobj = cifar_dataobj.map(operations=self.transform,input_columns='image')
        if self.train == "train":
            cifar_dataobj = cifar_dataobj.batch(50000)
        else:
            cifar_dataobj = cifar_dataobj.batch(10000)
        # data = data.batch(64)
        # print(target.get_dataset_size)
        data_iter = next(cifar_dataobj.create_dict_iterator())
        data = data_iter["image"].asnumpy()
        target = data_iter["label"].asnumpy().astype('int32')
        
        # train_tmp=[]
        # train_label_tmp = [] 
        # for data in cifar_dataobj.create_dict_iterator():
        #     img, label = data['image'],data['image']
        #     train_tmp.append(img.numpy())
        #     train_label_tmp.append(label)
        # data=np.array(train_tmp,dtype=np.uint8).reshape(-1,32,32,3)
        # label=np.array(train_label_tmp,dtype=np.uint8)

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

        # if self.transform is not None:
        #     img = self.transform(img)


        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(data.GeneratorDataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = Cifar100Dataset(self.root, self.train, self.transform, self.target_transform, self.download)

        # if torchvision.__version__ == '0.2.1':
        #     if self.train:
        #         data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
        #     else:
        #         data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        # else:
        data = cifar_dataobj.image
        target = np.array(cifar_dataobj.label)

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
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


# Iterable object as input source
class Iterable:
    def __init__(self,data,label):
        self._index = 0
        self._data = data
        self._label = label

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        else:
            item = (self._data[self._index], self._label[self._index])
            self._index += 1
            return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self._data)
