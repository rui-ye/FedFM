import paddle
import paddle.io as data
from PIL import Image
import numpy as np
import paddle.vision.transforms as transforms
from paddle.vision.datasets import Cifar10
from paddle.vision.datasets import Cifar100
from paddle.io import DataLoader,Dataset
from data_partition import partition_data_new

from paddle.vision.transforms import functional as F
from paddle.vision.transforms import BaseTransform

class ToPILImage(BaseTransform):
    def __init__(self, mode=None, keys=None):
        super(ToPILImage, self).__init__(keys)
        self.mode = mode

    def _apply_image(self, pic):
        """
        Args:
            pic (Tensor|np.ndarray): Image to be converted to PIL Image.
        Returns:
            PIL: Converted image.
        """
        if not (isinstance(pic, paddle.Tensor) or isinstance(pic, np.ndarray)):
            raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(
                type(pic)))

        elif isinstance(pic, paddle.Tensor):
            if pic.ndimension() not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndimension()))

            elif pic.ndimension() == 2:
                # if 2D image, add channel dimension (CHW)
                pic = pic.unsqueeze(0)

        elif isinstance(pic, np.ndarray):
            if pic.ndim not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndim))

            elif pic.ndim == 2:
                # if 2D image, add channel dimension (HWC)
                pic = np.expand_dims(pic, 2)

        npimg = pic
        if isinstance(pic, paddle.Tensor) and "float" in str(pic.numpy(
        ).dtype) and self.mode != 'F':
            pic = pic.mul(255).byte()
        if isinstance(pic, paddle.Tensor):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))

        if not isinstance(npimg, np.ndarray):
            raise TypeError(
                'Input pic must be a paddle.Tensor or NumPy ndarray, ' +
                'not {}'.format(type(npimg)))

        if npimg.shape[2] == 1:
            expected_mode = None
            npimg = npimg[:, :, 0]
            if npimg.dtype == np.uint8:
                expected_mode = 'L'
            elif npimg.dtype == np.int16:
                expected_mode = 'I;16'
            elif npimg.dtype == np.int32:
                expected_mode = 'I'
            elif npimg.dtype == np.float32:
                expected_mode = 'F'
            if self.mode is not None and self.mode != expected_mode:
                raise ValueError(
                    "Incorrect mode ({}) supplied for input type {}. Should be {}"
                    .format(self.mode, np.dtype, expected_mode))
            self.mode = expected_mode

        elif npimg.shape[2] == 2:
            permitted_2_channel_modes = ['LA']
            if self.mode is not None and self.mode not in permitted_2_channel_modes:
                raise ValueError("Only modes {} are supported for 2D inputs".
                                 format(permitted_2_channel_modes))

            if self.mode is None and npimg.dtype == np.uint8:
                self.mode = 'LA'

        elif npimg.shape[2] == 4:
            permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
            if self.mode is not None and self.mode not in permitted_4_channel_modes:
                raise ValueError("Only modes {} are supported for 4D inputs".
                                 format(permitted_4_channel_modes))

            if self.mode is None and npimg.dtype == np.uint8:
                self.mode = 'RGBA'
        else:
            permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
            if self.mode is not None and self.mode not in permitted_3_channel_modes:
                raise ValueError("Only modes {} are supported for 3D inputs".
                                 format(permitted_3_channel_modes))
            if self.mode is None and npimg.dtype == np.uint8:
                self.mode = 'RGB'

        if self.mode is None:
            raise TypeError('Input type {} is not supported'.format(
                npimg.dtype))

        return Image.fromarray(npimg, mode=self.mode)

class Cifar_Truncated(data.Dataset):
    def __init__(self, data, labels, transform=None):
        super(Cifar_Truncated, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)
    
def cifar_dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class,logger):
    if dataset == "cifar10":
        train_dataset = Cifar10(data_file="/GPFS/rhome/xinyuzhu/.cache/paddle/dataset/cifar/cifar-10-python.tar.gz",mode="train",download=False)
        test_dataset = Cifar10(data_file="/GPFS/rhome/xinyuzhu/.cache/paddle/dataset/cifar/cifar-10-python.tar.gz",mode="test",download=False)
        transform_train=transforms.Compose([
            ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform_test=transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset == "cifar100":
        train_dataset = Cifar100(base_path, True)
        test_dataset = Cifar100(base_path, False)
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train=transforms.Compose([
            ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        transform_test=transforms.Compose(
            [transforms.ToTensor(),
            normalize])
    train_tmp=[]
    train_label_tmp = []
    for i in range(50000):  
        img, label = train_dataset[i]
        train_tmp.append(img.getdata())
        train_label_tmp.append(label)
    train_image=np.array(train_tmp,dtype=np.uint8).reshape(-1,32,32,3)
    train_label=np.array(train_label_tmp,dtype=np.uint8)
    # train_image = train_dataset.data
    # train_label = np.array(train_dataset.targets)
    test_tmp=[]
    test_label_tmp = []
    for i in range(10000):  
        img, label = train_dataset[i]
        test_tmp.append(img.getdata())
        test_label_tmp.append(label)
    test_image=np.array(test_tmp,dtype=np.uint8).reshape(-1,32,32,3)
    test_label=np.array(test_label_tmp,dtype=np.uint8)
    # test_image = test_dataset.data
    # test_label = np.array(test_dataset.targets)
    # print(test_label)
    # print(test_image)
    n_train = train_label.shape[0]
    net_dataidx_map, traindata_cls_counts, data_distributions = partition_data_new(partition, n_train, n_parties, train_label ,logger,beta, skew_class)
    
    train_dataloaders = []
    val_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i][:int(0.8*len(net_dataidx_map[i]))]
        val_idxs = net_dataidx_map[i][int(0.8*len(net_dataidx_map[i])):]
        # print(train_idxs)
        # print(val_idxs)
        train_dataset = Cifar_Truncated(data=train_image[train_idxs], labels=train_label[train_idxs], transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = Cifar_Truncated(data=train_image[val_idxs], labels=train_label[val_idxs], transform=transform_test)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        train_dataloaders.append(train_loader)
        val_dataloaders.append(val_loader)
    
    test_dataset = Cifar_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, val_dataloaders, test_loader, net_dataidx_map, traindata_cls_counts, data_distributions
    