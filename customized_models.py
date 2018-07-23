import torch, torchvision
import torch.nn as nn
import torch.nn.init as init

class ResNet152(nn.Module):
    """
    Model modified.

    The architecture of our model is the same as standard Resnet152
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size, alternative_initialization = False):
        super().__init__()
        try:
            self.resnet152 = torchvision.models.resnet152(pretrained=True)
        except:
            print('server problem encoutered. setting pretrained = False')
            self.resnet152 = torchvision.models.resnet152(pretrained=False)
        num_ftrs = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

        if alternative_initialization:
            for module_ in self.modules():
                if isinstance(module_, nn.Conv2d):
                    init.xavier_uniform_(module_.weight.data)

                elif isinstance(module_, nn.Linear):
                    init.xavier_uniform_(module_.weight.data)
                    if module_.bias is not None:
                        init.constant(module_.bias, 0.0)

    def forward(self, x):
        x = self.resnet152(x)
        return x

class ResNet50(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard Resnet50
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size, alternative_initialization = False):
        super().__init__()
        try:
            self.resnet50 = torchvision.models.resnet50(pretrained=True)
        except:
            print('server problem encoutered. setting pretrained = False')
            self.resnet152 = torchvision.models.resnet152(pretrained=False)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

        if alternative_initialization:
            for module_ in self.modules():
                if isinstance(module_, nn.Conv2d):
                    init.xavier_uniform_(module_.weight.data)

                elif isinstance(module_, nn.Linear):
                    init.xavier_uniform_(module_.weight.data)
                    if module_.bias is not None:
                        init.constant_(module_.bias, 0.0)

    def forward(self, x):
        x = self.resnet50(x)
        return x

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size, alternative_initialization = False):
        super().__init__()
        try:
            self.densenet121 = torchvision.models.densenet121(pretrained=True)
        except:
            print('server problem encoutered. setting pretrained = False')
            self.densenet121 = torchvision.models.densenet121(pretrained=False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
            )

        if alternative_initialization:
            for module_ in self.modules():
                if isinstance(module_, nn.Conv2d):
                    init.xavier_uniform_(module_.weight.data)

                elif isinstance(module_, nn.Linear):
                    init.xavier_uniform_(module_.weight.data)
                    if module_.bias is not None:
                        init.constant(module_.bias, 0.0)

    def forward(self, x):
        x = self.densenet121(x)
        return x

class InceptionV3(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size, alternative_initialization = False):
        super().__init__()

        self.inception3 = torchvision.models.Inception3()
        num_ftrs = self.inception3.fc.in_features
        self.inception3.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

        if alternative_initialization:
            for module_ in self.modules():
                if isinstance(module_, nn.Conv2d):
                    init.xavier_uniform_(module_.weight.data)

                elif isinstance(module_, nn.Linear):
                    init.xavier_uniform_(module_.weight.data)
                    if module_.bias is not None:
                        init.constant(module_.bias, 0.0)

    def forward(self, x):
        x = self.inception3(x)
        return x
