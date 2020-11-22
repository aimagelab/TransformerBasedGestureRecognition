import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


__all__ = [
    'VGG', 'vgg16', 'vgg16_bn',
]


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def _vgg(arch, batch_norm, pretrained, in_planes, drop_prob, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg, batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=True)
        model.load_state_dict(state_dict)

        if in_planes in [1, 2]:
            w = model.features._modules['0']._parameters['weight'].data
            model.features._modules['0'] = nn.Conv2d(in_planes, 64, kernel_size=3, padding=1, bias=batch_norm is False)
            if in_planes == 1:
                model.conv1._parameters['weight'].data = w.mean(dim=1, keepdim=True)
            else:
                model.conv1._parameters['weight'].data = w[:, :-1] * 1.5
    else:
        model.features._modules['0'] = nn.Conv2d(in_planes, 64, kernel_size=3, padding=1, bias=batch_norm is False)
    if drop_prob > 0:
        new_features = list()
        for el in model.features:
            new_features.append(el)
            if isinstance(el, nn.ReLU):
                new_features.append(nn.Dropout2d(p=drop_prob))
        model.features = nn.Sequential(*new_features)

    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = None
    return model


def vgg16(pretrained=False, in_planes: int=3, dropout2d: float=0., **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', False, pretrained, in_planes, dropout2d, **kwargs)


def vgg16_bn(pretrained=False, in_planes: int=3, dropout2d: float=0., **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', True, pretrained, in_planes, dropout2d, **kwargs)