import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

class UpSample_EffNet(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSample_EffNet, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(output_features),
                                  nn.LeakyReLU())
    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self.net(f)


class Decoder_EffNet(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(Decoder_EffNet, self).__init__()
        features = int(num_features)
        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)
        self.up1 = UpSample_EffNet(skip_input=features // 1 + 112 + 64, output_features=features // 2 )
        self.up2 = UpSample_EffNet(skip_input=features // 2 + 40  + 24, output_features=features // 4 )
        self.up3 = UpSample_EffNet(skip_input=features // 4 + 24  + 16, output_features=features // 8 )
        self.up4 = UpSample_EffNet(skip_input=features // 8 + 16  + 8, output_features=features  // 16)

        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        #         x_d5 = self.up5(x_d4, features[0])
        #print(x_d4.shape)
        out = self.conv3(x_d4)

        return out

class Encoder_EffNet(nn.Module):
    def __init__(self, backend):
        super(Encoder_EffNet, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', "tf_efficientnet_b5_ap", pretrained=True, verbose=False)
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.encoder    = Encoder_EffNet(basemodel)
        self.classifier = nn.Sequential(nn.Flatten(1), nn.Linear(2048 * 8 * 8, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, 1))
    def forward(self, x): 
        out = self.encoder(x)[-1]
        out = F.interpolate(out, (8, 8), mode='bilinear', align_corners=True)
        return torch.sigmoid(self.classifier(out))
        
    def get_parameters(self):
        modules = [self.encoder, self.classifier]
        for m in modules:
            yield from m.parameters()
        

class Generator(nn.Module):
    def __init__(self, channels = 1):
        super(Generator, self).__init__()
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', "tf_efficientnet_b5_ap", pretrained=True, verbose=False)
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.encoder    = Encoder_EffNet(basemodel)
        self.decoder    = Decoder_EffNet(num_classes=channels)
    def forward(self, x): 
        out = self.encoder(x)
        return torch.relu(self.decoder(out))
    def get_parameters(self):
        modules = [self.encoder, self.decoder]
        for m in modules:
            yield from m.parameters()
        