import itertools
import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10

cifar10 = Cifar10()
print(len(cifar10))

for i in range(5):  # only show first 5 images
    img, label = cifar10[i]
    # do something with img and label
    print(type(img), img.size, label)
    # <class 'PIL.Image.Image'> (32, 32) 6


transform = T.Compose(
    [
        T.Resize(64),
        T.ToTensor(),
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            to_rgb=True,
        ),
    ]
)
cifar10_test = Cifar10(
    mode="test",
    transform=transform,  # apply transform to every image
    backend="cv2",  # use OpenCV as image transform backend
)
print(len(cifar10_test))

for img, label in itertools.islice(iter(cifar10_test), 5):  # only show first 5 images
    # do something with img and label
    print(type(img), img.shape, label)
    # <class 'paddle.Tensor'> [3, 64, 64] 3
