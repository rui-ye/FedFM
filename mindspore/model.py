from cgitb import reset
import mindspore
import mindspore.nn as nn
# import torch.nn.functional as F
import math
# import torchvision.models as models
# from resnetcifar import ResNet18_cifar10, ResNet50_cifar10
import sys
from resnet import resnet50,resnet18

#import pytorch_lightning as pl

class ModelFedCon_noheader(nn.Cell):

    def __init__(self, base_model, out_dim, n_classes, net_configs=None):
        super(ModelFedCon_noheader, self).__init__()
        print('no header')
        if base_model == "resnet50":
            basemodel = resnet50(10)
            basemodel.conv1 = mindspore.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            basemodel.maxpool = mindspore.nn.Identity()
            self.features = nn.SequentialCell(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet50_7":
            basemodel = resnet50(10)
            self.features = nn.SequentialCell(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18":
            basemodel = resnet18(10)
            basemodel.conv1 = mindspore.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            basemodel.maxpool = mindspore.nn.Identity()
            self.features = nn.SequentialCell(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18_7":
            basemodel = resnet18(10)
            self.features = nn.SequentialCell(*list(basemodel.cells())[:-1])
            # print('self.features',self.features)
            # num_ftrs = basemodel.fc.in_features
            num_ftrs = 512
            # print('number_ftrs',num_ftrs)
        elif base_model == 'resnet18_7_gn':
            basemodel = resnet18(pretrained=False)
            # Change BN to GN 
            basemodel.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            basemodel.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            basemodel.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            # assert len(dict(basemodel.named_parameters()).keys()) == len(basemodel.state_dict().keys()), 'More BN layers are there...'
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == 'resnet18_gn':
            basemodel = resnet18(pretrained=False)
            basemodel.conv1=mindspore.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
            basemodel.maxpool=mindspore.nn.Identity()
            # Change BN to GN 
            basemodel.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            basemodel.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            basemodel.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            basemodel.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            # assert len(dict(basemodel.named_parameters()).keys()) == len(basemodel.state_dict().keys()), 'More BN layers are there...'
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == 'resnet50_gn':
            basemodel = resnet50(pretrained=False)
            basemodel.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
            basemodel.maxpool=mindspore.nn.Identity()
            basemodel.bn1=nn.GroupNorm(num_groups = 2, num_channels = 64)
            
            basemodel.layer1[0].bn1=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn2=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[0].bn3=nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer1[0].downsample[1]=nn.GroupNorm(num_groups = 2, num_channels = 256)
            
            basemodel.layer1[1].bn1=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn2=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[1].bn3=nn.GroupNorm(num_groups = 2, num_channels = 256)
            
            basemodel.layer1[2].bn1=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[2].bn2=nn.GroupNorm(num_groups = 2, num_channels = 64)
            basemodel.layer1[2].bn3=nn.GroupNorm(num_groups = 2, num_channels = 256)
            
            basemodel.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[0].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            
            basemodel.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[1].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer2[2].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[2].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[2].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer2[3].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[3].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            basemodel.layer2[3].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            
            basemodel.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[0].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            
            basemodel.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[1].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[2].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[2].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[2].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[3].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[3].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[3].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[4].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[4].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[4].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            basemodel.layer3[5].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[5].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            basemodel.layer3[5].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 1024)
            
            basemodel.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[0].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 2048)
            basemodel.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 2048)
            
            basemodel.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[1].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 2048)
            basemodel.layer4[2].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[2].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            basemodel.layer4[2].bn3 = nn.GroupNorm(num_groups = 2, num_channels = 2048)
            assert len(dict(basemodel.named_parameters()).keys()) == len(basemodel.state_dict().keys()), 'More BN layers are there...'

            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features

        #summary(self.features.to('cuda:0'), (3,32,32))
        #print("features:", self.features)
        # projection MLP
        # self.l1 = nn.Linear(num_ftrs, num_ftrs)
        # self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Dense(num_ftrs, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def construct(self, x):
        h = self.features(x)
        #print("h before:", h)
        #print("h size:", h.size())
        h = h.squeeze()
        #print("h after:", h)
        # x = self.l1(h)
        # x = F.relu(x)
        # x = self.l2(x)

        y = self.l3(h)
        return h, h, y


if __name__=='__main__':
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for p in model.parameters():
            print(type(p[0]))
        return total_num, trainable_num
    model = ModelFedCon_noheader('simple-cnn', out_dim=0, n_classes=10, net_configs=None)
    print(get_parameter_number(model))