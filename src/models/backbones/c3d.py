import torch
import torch.nn as nn


class _C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, drop_prob: float):
        super(_C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.dropout = nn.Dropout3d(p=drop_prob)

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.dropout(h)
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.dropout(h)
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.dropout(h)
        h = self.relu(self.conv3b(h))
        h = self.dropout(h)
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.dropout(h)
        h = self.relu(self.conv4b(h))
        h = self.dropout(h)
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.dropout(h)
        h = self.relu(self.conv5b(h))
        h = self.dropout(h)
        h = self.pool5(h)

        h = self.avgpool(h)

        return h.squeeze()


def C3D(pretrained, in_planes: int=3, dropout=0., **kwargs):
    model = _C3D(drop_prob=dropout)
    if pretrained:
        state_dict = torch.load("./c3d.pickle")
        model.load_state_dict(state_dict)
        if in_planes in [1, 2]:
            w = model.conv1._parameters['weight'].data
            model.conv1 = nn.Conv3d(in_planes, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            if in_planes == 1:
                model.conv1._parameters['weight'].data = w.mean(dim=1, keepdim=True)
            else:
                model.conv1._parameters['weight'].data = w[:, :-1] * 1.5
    model.conv1 = nn.Conv3d(in_planes, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
    model.fc6 = None
    model.fc7 = None
    model.fc8 = None
    return model