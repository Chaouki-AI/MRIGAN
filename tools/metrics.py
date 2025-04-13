import torch
import numpy as np
from PIL import Image
import torch.nn as nn

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg

class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse)

class Loss(nn.Module): 
    def __init__(self, args):
        super(Loss, self).__init__()
        self.name = args.loss 
        self.L1 = nn.L1Loss() if args.loss == 'L1' else None
        
    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)    
        if mask is None:
            mask = target > 0.
            input  = input[mask]  + 1e-8
            target = target[mask] + 1e-8
        if self.name == "L1": 
            return self.L1(input, target)
        else : 
            g = torch.log(input) - torch.log(target)
            Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
            return torch.sqrt(Dg)


import torch
from torch import nn
from torchvision.models.vgg import vgg16
from torchvision.models import resnet50, ResNet50_Weights


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        #vgg = vgg16(pretrained=True)
        #loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        loss_network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        loss_network = nn.Sequential(*list(loss_network.children())[:-2]).eval()

        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.L1 = nn.L1Loss()

    def forward(self, out_labels, out_images, target_images):
        #print(out_labels, out_images.shape, target_images.shape)
        out_images = nn.functional.interpolate(out_images, size=[target_images.shape[2], target_images.shape[3]], mode='bilinear', align_corners=True)
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        self.loss_network = self.loss_network.to(out_images.device)
        #s = self.loss_network(out_images)
        #print(s.shape)
        perception_loss = self.l1_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.l1_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss
    
    def silog_loss(self, out_images, target_images):
        mask = target_images > 0.
        input  = out_images[mask]  + 1e-8
        target = target_images[mask] + 1e-8
        g = torch.log(input) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)
    
    def l1_loss(self, out_images, target_images):
        return self.L1(out_images, target_images)


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
