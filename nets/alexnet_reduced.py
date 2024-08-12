import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet_reduced', 'alexnet_reduced']


# model_urls = {
#     'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }

class AlexNet_reduced(nn.Module):

    def __init__(self):
        super(AlexNet_reduced, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet_reduced(reducing_rate=1, pretrained=False, progress=True, num_classes=1000):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet_reduced()
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['alexnet'], progress=progress)
        model.load_state_dict(state_dict)
    if num_classes != 1000:
        num_in_feature = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_in_feature, num_classes)
    return model