from typing import Any, Union

from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
}


class MyResNet(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18(pretrained: bool = False, **kwargs: Any) -> MyResNet:
    model = MyResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["resnet18"], progress=True)
        model.load_state_dict(state_dict)
        print("Starting from ImageNet weights...")
    return model


def resnet50(pretrained: bool = False, **kwargs: Any) -> MyResNet:
    model = MyResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["resnet50"], progress=True)
        model.load_state_dict(state_dict)
        print("Starting from ImageNet weights...")
    return model
