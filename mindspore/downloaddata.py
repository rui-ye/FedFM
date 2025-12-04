from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"

download(url, "./data/datasets-cifar10-bin", kind="tar.gz", replace=True)
