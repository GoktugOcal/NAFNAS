import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):

    def __init__(self,reducing_rate = 1):
        super(AlexNet, self).__init__()
        self.reducing_rate = reducing_rate
        self.features_rate = 1

        self.features = nn.Sequential(
            nn.Conv2d(3, int(64*self.features_rate), kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(int(64*self.features_rate), int(192*self.features_rate), kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(int(192*self.features_rate), int(384*self.features_rate), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(384*self.features_rate), int(256*self.features_rate), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(256*self.features_rate), int(256*self.features_rate), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(int(256 * 6 * 6*self.features_rate), int(4096*self.reducing_rate)),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(int(4096*self.features_rate), int(4096*self.reducing_rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(4096*self.reducing_rate), 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(reducing_rate=1, pretrained=False, progress=True, num_classes=1000):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(reducing_rate=reducing_rate)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['alexnet'], progress=progress)
        model.load_state_dict(state_dict)
    if num_classes != 1000:
        num_in_feature = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_in_feature, num_classes)
    return model